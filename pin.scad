// Units in inches
inch = 25.4;

// Pin dimensions
pin_diameter = 0.5 * inch;
pin_length = 6.5 * inch;

// Head dimensions
head_diameter = 0.75 * inch;
head_height = 0.2 * inch;

$fn = 100; // smooth cylinder resolution

// Final pin with head
union() {
    // Pin body
    cylinder(h = pin_length, d = pin_diameter, center = false);
    
    // Head on top
    translate([0, 0, pin_length])
        cylinder(h = head_height, d = head_diameter, center = false);
B
B
B
B
B
B
B
B
B
B
B
B
B
B
A
A
A
A
A
A
A
A
A
A
A
A
}
