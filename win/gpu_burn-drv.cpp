/*
 * Copyright (c) 2022, Ville Timonen
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *	this list of conditions and the following disclaimer in the
 *documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 *FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are
 *those of the authors and should not be interpreted as representing official
 *policies, either expressed or implied, of the FreeBSD Project.
 *
 * Windows Port: Modified for Windows compatibility
 */

// Windows-specific includes - must come before SIZE definition to avoid conflicts
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX  // Prevent Windows.h from defining min/max macros
#include <windows.h>
#include <io.h>
#include <process.h>
#include <signal.h>

// Define Windows compatibility types
#ifndef ssize_t
#ifdef _WIN64
typedef __int64 ssize_t;
#else
typedef int ssize_t;
#endif
#endif

// Define missing error codes
#ifndef ENOMEDIUM
#define ENOMEDIUM 123
#endif
#ifndef EMEDIUMTYPE
#define EMEDIUMTYPE 124
#endif

// Suppress MSVC warnings
#define _CRT_SECURE_NO_WARNINGS

// Matrices are MATRIX_SIZE*MATRIX_SIZE..  POT should be efficiently implemented in CUBLAS
// Using MATRIX_SIZE instead of SIZE to avoid Windows header conflicts
#define MATRIX_SIZE 8192ul
#define USEMEM 0.9 // Try to allocate 90% of memory
#define COMPARE_KERNEL "compare.fatbin"

// Used to report op/s, measured through Visual Profiler, CUBLAS from CUDA 7.5
// (Seems that they indeed take the naive dim^3 approach)
//#define OPS_PER_MUL 17188257792ul // Measured for MATRIX_SIZE = 2048
#define OPS_PER_MUL 1100048498688ul // Extrapolated for MATRIX_SIZE = 8192

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <errno.h>
#include <exception>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string.h>
#include <string>
#include <thread>
#include <time.h>
#include <vector>
#include <regex>

#define SIGTERM_TIMEOUT_THRESHOLD_SECS 30 // number of seconds for sigterm to kill child processes before forcing a sigkill

#include "cublas_v2.h"
#define CUDA_ENABLE_DEPRECATED
#include <cuda.h>

void _checkError(int rCode, std::string file, int line, std::string desc = "") {
    if (rCode != CUDA_SUCCESS) {
        const char *err;
        cuGetErrorString((CUresult)rCode, &err);

        throw std::runtime_error(
            (desc == "" ? std::string("Error (")
                        : (std::string("Error in ") + desc + " (")) +
            file + ":" + std::to_string(line) + "): " + err);
        // Yes, this *is* a memory leak, but this block is only executed on
        // error, so it's not a big deal
    }
}

void _checkError(cublasStatus_t rCode, std::string file, int line, std::string desc = "") {
    if (rCode != CUBLAS_STATUS_SUCCESS) {
#if CUBLAS_VER_MAJOR >= 12
		const char *err = cublasGetStatusString(rCode);
#else
		const char *err = "";
#endif
        throw std::runtime_error(
            (desc == "" ? std::string("Error (")
                        : (std::string("Error in ") + desc + " (")) +
            file + ":" + std::to_string(line) + "): " + err);
        // Yes, this *is* a memory leak, but this block is only executed on
        // error, so it's not a big deal
    }
}

#define checkError(rCode, ...)                                                 \
    _checkError(rCode, __FILE__, __LINE__, ##__VA_ARGS__)

double getTime() {
    FILETIME ft;
    ULARGE_INTEGER uli;
    GetSystemTimeAsFileTime(&ft);
    uli.LowPart = ft.dwLowDateTime;
    uli.HighPart = ft.dwHighDateTime;
    return (double)uli.QuadPart / 10000000.0 - 11644473600.0;
}

volatile bool g_running = false;

template <class T> class GPU_Test {
  public:
    GPU_Test(int dev, bool doubles, bool tensors, const char *kernelFile)
        : d_devNumber(dev), d_doubles(doubles), d_tensors(tensors), d_kernelFile(kernelFile){
        checkError(cuDeviceGet(&d_dev, d_devNumber));
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000
            checkError(cuCtxCreate(&d_ctx, nullptr, 0, d_dev));
#else
            checkError(cuCtxCreate(&d_ctx, 0, d_dev));
#endif

        bind();

        // checkError(cublasInit());
        checkError(cublasCreate(&d_cublas), "init");

        if (d_tensors)
            checkError(cublasSetMathMode(d_cublas, CUBLAS_TENSOR_OP_MATH));

        checkError(cuMemAllocHost((void **)&d_faultyElemsHost, sizeof(int)));
        d_error = 0;

        g_running = true;
    }
    ~GPU_Test() {
        bind();
        checkError(cuMemFree(d_Cdata), "Free A");
        checkError(cuMemFree(d_Adata), "Free B");
        checkError(cuMemFree(d_Bdata), "Free C");
        checkError(cuMemFree(d_faultyElemData), "Free faulty data");
        cuMemFreeHost(d_faultyElemsHost);
        printf("Freed memory for dev %d\n", d_devNumber);

        cublasDestroy(d_cublas);
        printf("Uninitted cublas\n");
    }

    static void termHandler(int signum) { g_running = false; }

    unsigned long long int getErrors() {
        if (*d_faultyElemsHost) {
            d_error += (long long int)*d_faultyElemsHost;
        }
        unsigned long long int tempErrs = d_error;
        d_error = 0;
        return tempErrs;
    }

    size_t getIters() { return d_iters; }

    void bind() { checkError(cuCtxSetCurrent(d_ctx), "Bind CTX"); }

    size_t totalMemory() {
        bind();
        size_t freeMem, totalMem;
        checkError(cuMemGetInfo(&freeMem, &totalMem));
        return totalMem;
    }

    size_t availMemory() {
        bind();
        size_t freeMem, totalMem;
        checkError(cuMemGetInfo(&freeMem, &totalMem));
        return freeMem;
    }

    void initBuffers(T *A, T *B, ssize_t useBytes = 0) {
        bind();

        if (useBytes == 0)
            useBytes = (ssize_t)((double)availMemory() * USEMEM);
        if (useBytes < 0)
            useBytes = (ssize_t)((double)availMemory() * (-useBytes / 100.0));

        printf("Initialized device %d with %zu MB of memory (%zu MB available, "
               "using %lld MB of it), %s%s\n",
               d_devNumber, totalMemory() / 1024ul / 1024ul,
               availMemory() / 1024ul / 1024ul, (long long)(useBytes / 1024ul / 1024ul),
               d_doubles ? "using DOUBLES" : "using FLOATS",
               d_tensors ? ", using Tensor Cores" : "");
        size_t d_resultSize = sizeof(T) * MATRIX_SIZE * MATRIX_SIZE;
        d_iters = (useBytes - 2 * d_resultSize) /
                  d_resultSize; // We remove A and B sizes
        printf("Results are %zu bytes each, thus performing %zu iterations\n",
               d_resultSize, d_iters);
        if ((size_t)useBytes < 3 * d_resultSize)
            throw std::string("Low mem for result. aborting.\n");
        checkError(cuMemAlloc(&d_Cdata, d_iters * d_resultSize), "C alloc");
        checkError(cuMemAlloc(&d_Adata, d_resultSize), "A alloc");
        checkError(cuMemAlloc(&d_Bdata, d_resultSize), "B alloc");

        checkError(cuMemAlloc(&d_faultyElemData, sizeof(int)), "faulty data");

        // Populating matrices A and B
        checkError(cuMemcpyHtoD(d_Adata, A, d_resultSize), "A -> device");
        checkError(cuMemcpyHtoD(d_Bdata, B, d_resultSize), "B -> device");

        initCompareKernel();
    }

    void compute() {
        bind();
        static const float alpha = 1.0f;
        static const float beta = 0.0f;
        static const double alphaD = 1.0;
        static const double betaD = 0.0;

        for (size_t i = 0; i < d_iters; ++i) {
            if (d_doubles)
                checkError(
                    cublasDgemm(d_cublas, CUBLAS_OP_N, CUBLAS_OP_N, MATRIX_SIZE, MATRIX_SIZE,
                                MATRIX_SIZE, &alphaD, (const double *)d_Adata, MATRIX_SIZE,
                                (const double *)d_Bdata, MATRIX_SIZE, &betaD,
                                (double *)d_Cdata + i * MATRIX_SIZE * MATRIX_SIZE, MATRIX_SIZE),
                    "DGEMM");
            else
                checkError(
                    cublasSgemm(d_cublas, CUBLAS_OP_N, CUBLAS_OP_N, MATRIX_SIZE, MATRIX_SIZE,
                                MATRIX_SIZE, &alpha, (const float *)d_Adata, MATRIX_SIZE,
                                (const float *)d_Bdata, MATRIX_SIZE, &beta,
                                (float *)d_Cdata + i * MATRIX_SIZE * MATRIX_SIZE, MATRIX_SIZE),
                    "SGEMM");
        }
    }

    void initCompareKernel() {
        {
            std::ifstream f(d_kernelFile);
            checkError(f.good() ? CUDA_SUCCESS : CUDA_ERROR_NOT_FOUND,
                       std::string("couldn't find compare kernel: ") + d_kernelFile);
        }
        checkError(cuModuleLoad(&d_module, d_kernelFile), "load module");
        checkError(cuModuleGetFunction(&d_function, d_module,
                                       d_doubles ? "compareD" : "compare"),
                   "get func");

        checkError(cuFuncSetCacheConfig(d_function, CU_FUNC_CACHE_PREFER_L1),
                   "L1 config");
        checkError(cuParamSetSize(d_function, __alignof(T *) +
                                                  __alignof(int *) +
                                                  __alignof(size_t)),
                   "set param size");
        checkError(cuParamSetv(d_function, 0, &d_Cdata, sizeof(T *)),
                   "set param");
        checkError(cuParamSetv(d_function, __alignof(T *), &d_faultyElemData,
                               sizeof(T *)),
                   "set param");
        checkError(cuParamSetv(d_function, __alignof(T *) + __alignof(int *),
                               &d_iters, sizeof(size_t)),
                   "set param");

        checkError(cuFuncSetBlockShape(d_function, g_blockSize, g_blockSize, 1),
                   "set block size");
    }

    void compare() {
        checkError(cuMemsetD32Async(d_faultyElemData, 0, 1, 0), "memset");
        checkError(cuLaunchGridAsync(d_function, MATRIX_SIZE / g_blockSize,
                                     MATRIX_SIZE / g_blockSize, 0),
                   "Launch grid");
        checkError(cuMemcpyDtoHAsync(d_faultyElemsHost, d_faultyElemData,
                                     sizeof(int), 0),
                   "Read faultyelemdata");
    }

    bool shouldRun() { return g_running; }

  private:
    bool d_doubles;
    bool d_tensors;
    int d_devNumber;
    const char *d_kernelFile;
    size_t d_iters;
    size_t d_resultSize;

    long long int d_error;

    static const int g_blockSize = 16;

    CUdevice d_dev;
    CUcontext d_ctx;
    CUmodule d_module;
    CUfunction d_function;

    CUdeviceptr d_Cdata;
    CUdeviceptr d_Adata;
    CUdeviceptr d_Bdata;
    CUdeviceptr d_faultyElemData;
    int *d_faultyElemsHost;

    cublasHandle_t d_cublas;
};

// Returns the number of devices
int initCuda() {
    try {
        CUresult initResult = cuInit(0);
        const char *initErrStr = "<unavailable>";
        if (cuGetErrorString(initResult, &initErrStr) != CUDA_SUCCESS ||
            initErrStr == nullptr) {
                initErrStr = "<unavailable>";
            }
        fprintf(stderr, "cuInit returned %d (%s)\n", initResult,
            initErrStr);
        checkError(initResult);
    } catch (std::runtime_error e) {
        fprintf(stderr, "Couldn't init CUDA: %s\n", e.what());
        return 0;
    }
    int deviceCount = 0;
    checkError(cuDeviceGetCount(&deviceCount));

    if (!deviceCount)
        throw std::string("No CUDA devices");

    return deviceCount;
}

template <class T>
unsigned int __stdcall startBurnThread(void *arg) {
    struct BurnThreadArgs {
        int index;
        HANDLE writePipe;
        T *A;
        T *B;
        bool doubles;
        bool tensors;
        ssize_t useBytes;
        const char *kernelFile;
    };
    
    BurnThreadArgs *args = (BurnThreadArgs *)arg;
    GPU_Test<T> *our;
    try {
        our = new GPU_Test<T>(args->index, args->doubles, args->tensors, args->kernelFile);
        our->initBuffers(args->A, args->B, args->useBytes);
        fflush(stdout);  // Ensure initialization messages are displayed immediately
    } catch (const std::exception &e) {
        fprintf(stderr, "Couldn't init a GPU test: %s\n", e.what());
        int ops = -1;
        DWORD written;
        WriteFile(args->writePipe, &ops, sizeof(int), &written, NULL);
        WriteFile(args->writePipe, &ops, sizeof(int), &written, NULL);
        CloseHandle(args->writePipe);
        delete args;
        _endthreadex(EMEDIUMTYPE);
        return EMEDIUMTYPE;
    }

    // The actual work
    try {
        int eventIndex = 0;
        const int maxEvents = 2;
        CUevent events[maxEvents];
        for (int i = 0; i < maxEvents; ++i)
            cuEventCreate(events + i, 0);

        int nonWorkIters = maxEvents;

        while (our->shouldRun()) {
            our->compute();
            our->compare();
            checkError(cuEventRecord(events[eventIndex], 0), "Record event");

            eventIndex = ++eventIndex % maxEvents;

            while (cuEventQuery(events[eventIndex]) != CUDA_SUCCESS)
                Sleep(1);

            if (--nonWorkIters > 0)
                continue;

            // Send progress update to main thread
            // Write the number of iterations processed in this batch
            int ops = (int)our->getIters();
            DWORD written;
            BOOL writeResult = WriteFile(args->writePipe, &ops, sizeof(int), &written, NULL);
            if (!writeResult) {
                DWORD error = GetLastError();
                // ERROR_BROKEN_PIPE (109) and ERROR_NO_DATA (232) are expected when reader closes
                // ERROR_PIPE_NOT_CONNECTED (535) can occur if pipe is disconnected
                if (error != ERROR_BROKEN_PIPE && error != ERROR_NO_DATA && error != ERROR_PIPE_NOT_CONNECTED) {
                    fprintf(stderr, "[GPU %d] Failed to write ops to pipe: error %lu\n", args->index, error);
                }
                // If pipe is broken, exit the loop
                if (error == ERROR_BROKEN_PIPE) {
                    break;
                }
            } else if (written != sizeof(int)) {
                fprintf(stderr, "[GPU %d] Partial write to pipe: wrote %lu of %zu bytes\n", args->index, written, sizeof(int));
            }
            
            ops = (int)our->getErrors();
            writeResult = WriteFile(args->writePipe, &ops, sizeof(int), &written, NULL);
            if (!writeResult) {
                DWORD error = GetLastError();
                if (error != ERROR_BROKEN_PIPE && error != ERROR_NO_DATA && error != ERROR_PIPE_NOT_CONNECTED) {
                    fprintf(stderr, "[GPU %d] Failed to write errors to pipe: error %lu\n", args->index, error);
                }
                // If pipe is broken, exit the loop
                if (error == ERROR_BROKEN_PIPE) {
                    break;
                }
            } else if (written != sizeof(int)) {
                fprintf(stderr, "[GPU %d] Partial write to pipe: wrote %lu of %zu bytes\n", args->index, written, sizeof(int));
            }
            
            // No FlushFileBuffers here, as it blocks until the reader reads the bytes,
            // which causes a deadlock when the main thread stops reading to wait for thread exit.
        }

        for (int i = 0; i < maxEvents; ++i)
            cuEventSynchronize(events[i]);
        delete our;
        
        // Close write pipe when thread is done
        CloseHandle(args->writePipe);
    } catch (const std::exception &e) {
        fprintf(stderr, "Failure during compute: %s\n", e.what());
        int ops = -1;
        DWORD written;
        // Signalling that we failed
        WriteFile(args->writePipe, &ops, sizeof(int), &written, NULL);
        WriteFile(args->writePipe, &ops, sizeof(int), &written, NULL);
        CloseHandle(args->writePipe);
        delete args;
        _endthreadex(ECONNREFUSED);
        return ECONNREFUSED;
    }
    
    delete args;
    _endthreadex(0);
    return 0;
}

HANDLE pollTemp(HANDLE *hProcess) {
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(SECURITY_ATTRIBUTES);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = NULL;

    HANDLE hReadPipe, hWritePipe;
    if (!CreatePipe(&hReadPipe, &hWritePipe, &sa, 0)) {
        return INVALID_HANDLE_VALUE;
    }

    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = hWritePipe;
    si.hStdError = hWritePipe;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

    char cmdLine[] = "nvidia-smi.exe -l 5 -q -d TEMPERATURE";
    if (!CreateProcessA(NULL, cmdLine, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
        CloseHandle(hReadPipe);
        CloseHandle(hWritePipe);
        return INVALID_HANDLE_VALUE;
    }

    CloseHandle(hWritePipe);
    *hProcess = pi.hProcess;
    CloseHandle(pi.hThread);

    return hReadPipe;
}

void updateTemps(HANDLE handle, std::vector<int> *temps) {
    const int readSize = 10240;
    static int gpuIter = 0;
    char data[readSize + 1];
    DWORD bytesRead;

    int curPos = 0;
    char ch;
    do {
        if (!ReadFile(handle, &ch, 1, &bytesRead, NULL) || bytesRead == 0)
            return;
        data[curPos++] = ch;
    } while (ch != '\n' && curPos < readSize);

    data[curPos - 1] = 0;

    // FIXME: The syntax of this print might change in the future..
    int tempValue;
    // Use sscanf_s - for integers, no buffer size needed
    if (sscanf_s(data,
               "		GPU Current Temp			: %d C",
               &tempValue) == 1) {
        temps->at(gpuIter) = tempValue;
        gpuIter = (gpuIter + 1) % (temps->size());
    } else if (!strcmp(data, "		Gpu				"
                             "	 : N/A"))
        gpuIter =
            (gpuIter + 1) %
            (temps->size()); // We rotate the iterator for N/A values as well
}

void listenClients(std::vector<HANDLE> clientHandles, std::vector<HANDLE> clientThreads,
                   int runTime, std::chrono::seconds sigterm_timeout_threshold_secs) {
    HANDLE tempProcess = NULL;
    HANDLE tempHandle = pollTemp(&tempProcess);
    if (tempHandle == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Could not start nvidia-smi for temperature monitoring\n");
    }

    std::vector<HANDLE> allHandles;
    // Only add tempHandle if it's valid
    if (tempHandle != INVALID_HANDLE_VALUE) {
        allHandles.push_back(tempHandle);
    }
    for (size_t i = 0; i < clientHandles.size(); ++i) {
        allHandles.push_back(clientHandles[i]);
    }

    std::vector<int> clientTemp;
    std::vector<int> clientErrors;
    std::vector<int> clientCalcs;
    std::vector<FILETIME> clientUpdateTime;
    std::vector<float> clientGflops;
    std::vector<bool> clientFaulty;
    std::vector<bool> clientFirstUpdate;

    time_t startTime = time(0);

    for (size_t i = 0; i < clientHandles.size(); ++i) {
        clientTemp.push_back(0);
        clientErrors.push_back(0);
        clientCalcs.push_back(0);
        FILETIME thisTime;
        GetSystemTimeAsFileTime(&thisTime);
        clientUpdateTime.push_back(thisTime);
        clientGflops.push_back(0.0f);
        clientFaulty.push_back(false);
        clientFirstUpdate.push_back(true);
    }

    float nextReport = 10.0f;
    bool childReport = false;
    bool hasTempHandle = (tempHandle != INVALID_HANDLE_VALUE);

    // Status line redraws on real events. Linux uses select() which wakes on
    // pipe data; we poll every 50ms and have to track that manually here.
    while (true) {
        time_t thisTime = time(0);
        if (startTime + runTime < thisTime)
            break;

        bool newData = false;
        // Poll all client pipes for data
        for (size_t i = 0; i < clientHandles.size(); ++i) {
            DWORD bytesAvailable = 0;
            if (PeekNamedPipe(clientHandles[i], NULL, 0, NULL, &bytesAvailable, NULL)) {
                // Need at least 2 ints (ops + errors)
                if (bytesAvailable >= 2 * sizeof(int)) {
                    FILETIME thisTimeSpec;
                    GetSystemTimeAsFileTime(&thisTimeSpec);
                    
                    int processed = 0, errors = 0;
                    DWORD bytesRead;
                    
                    // Read both ints atomically if possible
                    if (ReadFile(clientHandles[i], &processed, sizeof(int), &bytesRead, NULL) && bytesRead == sizeof(int)) {
                        if (ReadFile(clientHandles[i], &errors, sizeof(int), &bytesRead, NULL) && bytesRead == sizeof(int)) {
                            clientErrors.at(i) += errors;
                            if (processed == -1)
                                clientCalcs.at(i) = -1;
                            else {
                                FILETIME clientPrevTime = clientUpdateTime.at(i);
                                ULARGE_INTEGER prev, curr;
                                prev.LowPart = clientPrevTime.dwLowDateTime;
                                prev.HighPart = clientPrevTime.dwHighDateTime;
                                curr.LowPart = thisTimeSpec.dwLowDateTime;
                                curr.HighPart = thisTimeSpec.dwHighDateTime;
                                double clientTimeDelta = (double)(curr.QuadPart - prev.QuadPart) / 10000000.0;
                                
                                // Update time first
                                clientUpdateTime.at(i) = thisTimeSpec;
                                
                                // Calculate Gflops if we have a valid time delta and not first update
                                if (clientTimeDelta > 0.0 && !clientFirstUpdate.at(i)) {
                                    clientGflops.at(i) =
                                        (float)((double)((unsigned long long int)processed *
                                                 OPS_PER_MUL) /
                                        clientTimeDelta / 1000.0 / 1000.0 / 1000.0);
                                } else {
                                    // First update, set to 0 and mark as updated
                                    clientGflops.at(i) = 0.0f;
                                    clientFirstUpdate.at(i) = false;
                                }
                                
                                // Always update calc count
                                clientCalcs.at(i) += processed;
                            }
                            childReport = true;
                            newData = true;
                        } else {
                            // Failed to read errors, but we already read processed
                            // This shouldn't happen, but handle it
                        }
                    } else {
                        // Failed to read processed
                    }
                }
            } else {
                // PeekNamedPipe failed - check if pipe is broken
                DWORD error = GetLastError();
                if (error == ERROR_BROKEN_PIPE) {
                    // Thread may have exited
                    if (clientCalcs.at(i) != -1) {
                        clientCalcs.at(i) = -1;
                    }
                }
            }
        }
        
        // Check temperature handle if available
        if (hasTempHandle && tempHandle != INVALID_HANDLE_VALUE) {
            DWORD bytesAvailable = 0;
            if (PeekNamedPipe(tempHandle, NULL, 0, NULL, &bytesAvailable, NULL) && bytesAvailable > 0) {
                updateTemps(tempHandle, &clientTemp);
                // No newData here: nvidia-smi dumps ~50 lines per poll
                // cycle, and we drain one per iteration. Setting newData
                // would redraw the status line on every Sleep(50).
                // The new temp shows up on the next client report.
            }
        }

        // Only redraw the status line when something actually changed this
        // iteration; otherwise we'd spam ~20 identical lines per second.
        if (childReport && newData) {
            float elapsed = fminf((float)(thisTime - startTime) / (float)runTime * 100.0f, 100.0f);
            printf("\r%.1f%%  ", elapsed);
            fflush(stdout);
            printf("proc'd: ");
            for (size_t i = 0; i < clientCalcs.size(); ++i) {
                printf("%d (%.0f Gflop/s) ", clientCalcs.at(i), clientGflops.at(i));
                if (i != clientCalcs.size() - 1)
                    printf("- ");
            }
            printf("  errors: ");
            for (size_t i = 0; i < clientErrors.size(); ++i) {
                std::string note = "%d ";
                if (clientCalcs.at(i) == -1)
                    note += " (DIED!)";
                else if (clientErrors.at(i))
                    note += " (WARNING!)";
                printf(note.c_str(), clientErrors.at(i));
                if (i != clientErrors.size() - 1)
                    printf("- ");
            }
            printf("  temps: ");
            for (size_t i = 0; i < clientTemp.size(); ++i) {
                printf(clientTemp.at(i) != 0 ? "%d C " : "-- ", clientTemp.at(i));
                if (i != clientCalcs.size() - 1)
                    printf("- ");
            }
            fflush(stdout);
            
            for (size_t i = 0; i < clientErrors.size(); ++i)
                if (clientErrors.at(i))
                    clientFaulty.at(i) = true;

            if (nextReport < elapsed) {
                nextReport = elapsed + 10.0f;
                printf("\n\tSummary at:   ");
                fflush(stdout);
                SYSTEMTIME st;
                GetLocalTime(&st);
                printf("%04d-%02d-%02d %02d:%02d:%02d\n", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
                fflush(stdout);
                printf("\n");
                for (size_t i = 0; i < clientErrors.size(); ++i)
                    clientErrors.at(i) = 0;
            }
        }
        
        // Check if all clients are dead
        bool oneAlive = false;
        for (size_t i = 0; i < clientCalcs.size(); ++i)
            if (clientCalcs.at(i) != -1)
                oneAlive = true;
        if (!oneAlive) {
            fprintf(stderr, "\n\nNo clients are alive!  Aborting\n");
            exit(ENOMEDIUM);
        }
        
        // Sleep minimally to maintain responsiveness similar to Linux select()
        Sleep(50);
    }

    printf("\nTerminating threads\n");
    fflush(stdout);
    g_running = false;

    // Wait for all threads to finish up to sigterm_timeout_threshold_secs
    double timeoutSecs = (double)sigterm_timeout_threshold_secs.count();
    auto startWait = std::chrono::steady_clock::now();

    if (clientThreads.size() <= MAXIMUM_WAIT_OBJECTS) {
        DWORD timeoutMs = (DWORD)(timeoutSecs * 1000.0);
        if (timeoutMs < 0) timeoutMs = 0;
        WaitForMultipleObjects(
            (DWORD)clientThreads.size(),
            clientThreads.data(),
            TRUE, // Wait for ALL threads
            timeoutMs
        );
    } else {
        // Sequentially wait for each thread with remaining timeout
        for (size_t i = 0; i < clientThreads.size(); ++i) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - startWait).count();
            double remaining = timeoutSecs - elapsed;
            DWORD timeoutMs = (remaining > 0.0) ? (DWORD)(remaining * 1000.0) : 0;
            WaitForSingleObject(clientThreads[i], timeoutMs);
        }
    }

    // Force terminate any threads that are still alive after the wait
    for (size_t i = 0; i < clientThreads.size(); ++i) {
        DWORD exitCode;
        if (GetExitCodeThread(clientThreads[i], &exitCode) && exitCode == STILL_ACTIVE) {
            TerminateThread(clientThreads[i], 1);
        }
    }

    if (tempHandle != INVALID_HANDLE_VALUE) {
        TerminateProcess(tempProcess, 1);
        CloseHandle(tempHandle);
        CloseHandle(tempProcess);
    }

    for (size_t i = 0; i < clientHandles.size(); ++i) {
        CloseHandle(clientHandles[i]);
        CloseHandle(clientThreads[i]);
    }

    printf("done\n");

    printf("\nTested %d GPUs:\n", (int)clientThreads.size());
    for (size_t i = 0; i < clientThreads.size(); ++i)
        printf("\tGPU %d: %s\n", (int)i, clientFaulty.at(i) ? "FAULTY" : "OK");
}

template <class T>
void launch(int runLength, bool useDoubles, bool useTensorCores,
            ssize_t useBytes, int device_id, const char * kernelFile,
            std::chrono::seconds sigterm_timeout_threshold_secs) {
    system("nvidia-smi -L");
    fflush(stdout);

    // Initting A and B with random data
    T *A = (T *)malloc(sizeof(T) * MATRIX_SIZE * MATRIX_SIZE);
    T *B = (T *)malloc(sizeof(T) * MATRIX_SIZE * MATRIX_SIZE);
    srand(10);
    for (size_t i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i) {
        A[i] = (T)((double)(rand() % 1000000) / 100000.0);
        B[i] = (T)((double)(rand() % 1000000) / 100000.0);
    }

    std::vector<HANDLE> clientPipes;
    std::vector<HANDLE> clientThreads;

    if (device_id > -1) {
        SECURITY_ATTRIBUTES sa;
        sa.nLength = sizeof(SECURITY_ATTRIBUTES);
        sa.bInheritHandle = TRUE;
        sa.lpSecurityDescriptor = NULL;

        HANDLE hReadPipe, hWritePipe;
        // Create pipe with default buffer size
        if (!CreatePipe(&hReadPipe, &hWritePipe, &sa, 0)) {
            fprintf(stderr, "Failed to create pipe: error %lu\n", GetLastError());
            exit(1);
        }

        struct BurnThreadArgs {
            int index;
            HANDLE writePipe;
            T *A;
            T *B;
            bool doubles;
            bool tensors;
            ssize_t useBytes;
            const char *kernelFile;
        };

        BurnThreadArgs *args = new BurnThreadArgs;
        args->index = device_id;
        args->writePipe = hWritePipe;
        args->A = A;
        args->B = B;
        args->doubles = useDoubles;
        args->tensors = useTensorCores;
        args->useBytes = useBytes;
        args->kernelFile = kernelFile;

        HANDLE hThread = (HANDLE)_beginthreadex(NULL, 0, startBurnThread<T>, args, 0, NULL);
        if (hThread == NULL) {
            fprintf(stderr, "Failed to create thread\n");
            exit(1);
        }

        clientPipes.push_back(hReadPipe);
        clientThreads.push_back(hThread);
        // Don't close hWritePipe here - the thread needs it
        // The thread will close it when done

        listenClients(clientPipes, clientThreads, runLength, sigterm_timeout_threshold_secs);
        
        // Close write pipe after thread is done (it should be closed by thread, but just in case)
        CloseHandle(hWritePipe);
    } else {
        int devCount = initCuda();
        if (!devCount) {
            fprintf(stderr, "No CUDA devices\n");
            exit(ENODEV);
        }

        for (int i = 0; i < devCount; ++i) {
            SECURITY_ATTRIBUTES sa;
            sa.nLength = sizeof(SECURITY_ATTRIBUTES);
            sa.bInheritHandle = TRUE;
            sa.lpSecurityDescriptor = NULL;

            HANDLE hReadPipe, hWritePipe;
            if (!CreatePipe(&hReadPipe, &hWritePipe, &sa, 0)) {
                fprintf(stderr, "Failed to create pipe for GPU %d: error %lu\n", i, GetLastError());
                continue;
            }

            struct BurnThreadArgs {
                int index;
                HANDLE writePipe;
                T *A;
                T *B;
                bool doubles;
                bool tensors;
                ssize_t useBytes;
                const char *kernelFile;
            };

            BurnThreadArgs *args = new BurnThreadArgs;
            args->index = i;
            args->writePipe = hWritePipe;
            args->A = A;
            args->B = B;
            args->doubles = useDoubles;
            args->tensors = useTensorCores;
            args->useBytes = useBytes;
            args->kernelFile = kernelFile;

            HANDLE hThread = (HANDLE)_beginthreadex(NULL, 0, startBurnThread<T>, args, 0, NULL);
            if (hThread == NULL) {
                fprintf(stderr, "Failed to create thread for GPU %d\n", i);
                CloseHandle(hReadPipe);
                CloseHandle(hWritePipe);
                continue;
            }

            clientPipes.push_back(hReadPipe);
            clientThreads.push_back(hThread);
            // Don't close hWritePipe here - the thread needs it
            // The thread will close it when done
        }

        if (clientPipes.empty()) {
            fprintf(stderr, "Failed to create any GPU threads\n");
            exit(1);
        }

        listenClients(clientPipes, clientThreads, runLength, sigterm_timeout_threshold_secs);
        
        // Note: Write pipes are closed by their respective threads when they exit
    }

    free(A);
    free(B);
}

void showHelp() {
    printf("GPU Burn\n");
    printf("Usage: gpu-burn.exe [OPTIONS] [TIME]\n\n");
    printf("-m X\tUse X MB of memory.\n");
    printf("-m N%%\tUse N%% of the available GPU memory.  Default is %d%%\n",
           (int)(USEMEM * 100));
    printf("-d\tUse doubles\n");
    printf("-tc\tTry to use Tensor cores\n");
    printf("-l\tLists all GPUs in the system\n");
    printf("-i N\tExecute only on GPU N\n");
    printf("-c FILE\tUse FILE as compare kernel.  Default is %s\n",
           COMPARE_KERNEL);
    printf("-stts T\tSet timeout threshold to T seconds for using SIGTERM to abort child processes before using SIGKILL.  Default is %d\n",
           SIGTERM_TIMEOUT_THRESHOLD_SECS);
    printf("-h\tShow this help message\n\n");
    printf("Examples:\n");
    printf("  gpu-burn.exe -d 3600 # burns all GPUs with doubles for an hour\n");
    printf(
        "  gpu-burn.exe -m 50%% # burns using 50%% of the available GPU memory\n");
    printf("  gpu-burn.exe -l # list GPUs\n");
    printf("  gpu-burn.exe -i 2 # burns only GPU of index 2\n");
}

// NNN MB
// NN% <0
// 0 --- error
ssize_t decodeUSEMEM(const char *s) {
    char *s2;
    int64_t r = strtoll(s, &s2, 10);
    if (s == s2)
        return 0;
    if (*s2 == '%')
        return (s2[1] == 0) ? -r : 0;
    return (*s2 == 0) ? r * 1024 * 1024 : 0;
}

int main(int argc, char **argv) {
    int runLength = 10;
    bool useDoubles = false;
    bool useTensorCores = false;
    int thisParam = 0;
    ssize_t useBytes = 0; // 0 == use USEMEM% of free mem
    int device_id = -1;
    char *kernelFile = (char *)COMPARE_KERNEL;
    std::chrono::seconds sigterm_timeout_threshold_secs = std::chrono::seconds(SIGTERM_TIMEOUT_THRESHOLD_SECS);

    std::vector<std::string> args(argv, argv + argc);
    for (size_t i = 1; i < args.size(); ++i) {
        if (argc >= 2 && std::string(argv[i]).find("-h") != std::string::npos) {
            showHelp();
            return 0;
        }
        if (argc >= 2 && std::string(argv[i]).find("-l") != std::string::npos) {
            int count = initCuda();
            if (count == 0) {
                throw std::runtime_error("No CUDA capable GPUs found.\n");
            }
            for (int i_dev = 0; i_dev < count; i_dev++) {
                CUdevice device_l;
                char device_name[255];
                checkError(cuDeviceGet(&device_l, i_dev));
                checkError(cuDeviceGetName(device_name, 255, device_l));
                size_t device_mem_l;
                checkError(cuDeviceTotalMem(&device_mem_l, device_l));
                printf("ID %i: %s, %zuMB\n", i_dev, device_name,
                       device_mem_l / 1000 / 1000);
            }
            thisParam++;
            return 0;
        }
        if (argc >= 2 && std::string(argv[i]).find("-d") != std::string::npos) {
            useDoubles = true;
            thisParam++;
        }
        if (argc >= 2 &&
            std::string(argv[i]).find("-tc") != std::string::npos) {
            useTensorCores = true;
            thisParam++;
        }
        if (argc >= 2 && strncmp(argv[i], "-m", 2) == 0) {
            thisParam++;

            // -mNNN[%]
            // -m NNN[%]
            if (argv[i][2]) {
                useBytes = decodeUSEMEM(argv[i] + 2);
            } else if (i + 1 < args.size()) {
                i++;
                thisParam++;
                useBytes = decodeUSEMEM(argv[i]);
            } else {
                fprintf(stderr, "Syntax error near -m\n");
                exit(EINVAL);
            }
            if (useBytes == 0) {
                fprintf(stderr, "Syntax error near -m\n");
                exit(EINVAL);
            }
        }
        if (argc >= 2 && strncmp(argv[i], "-i", 2) == 0) {
            thisParam++;

            if (argv[i][2]) {
                device_id = strtol(argv[i] + 2, NULL, 0);
            } else if (i + 1 < args.size()) {
                i++;
                thisParam++;
                device_id = strtol(argv[i], NULL, 0);
            } else {
                fprintf(stderr, "Syntax error near -i\n");
                exit(EINVAL);
            }
        }
        if (argc >= 2 && strncmp(argv[i], "-c", 2) == 0) {
            thisParam++;

            if (argv[i + 1]) {
                kernelFile = argv[i + 1];
                thisParam++;
            }
        }
        if (argc >= 2 && strncmp(argv[i], "-stts", 2) == 0) {
            thisParam++;

            if (argv[i + 1]) {
                sigterm_timeout_threshold_secs = std::chrono::seconds(atoi(argv[i + 1]));
                thisParam++;
            }
        }
    }

    if (argc - thisParam < 2)
        printf("Run length not specified in the command line. ");
    else
        runLength = atoi(argv[1 + thisParam]);
    printf("Using compare file: %s\n", kernelFile);
    fflush(stdout);
    printf("Burning for %d seconds.\n", runLength);
    fflush(stdout);

    if (useDoubles)
        launch<double>(runLength, useDoubles, useTensorCores, useBytes,
                       device_id, kernelFile, sigterm_timeout_threshold_secs);
    else
        launch<float>(runLength, useDoubles, useTensorCores, useBytes,
                      device_id, kernelFile, sigterm_timeout_threshold_secs);

    return 0;
}

