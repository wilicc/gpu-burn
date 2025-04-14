// Units in inches
inch = 25.4;

// Dimensions
cube_length = 7 * inch;    // X
cube_breadth = 7 * inch;   // Z
cube_height = 1 * inch;    // Y

cyl_outer_d = 1 * inch;
cyl_inner_d = 0.5 * inch;
cyl_height = 2 * inch;

$fn = 100; // smooth curves

// Base cube
cube([cube_length, cube_height, cube_breadth]);

// First cylinder (top) - now correctly attached to left face (X=0)
translate([-7, cube_height /2 , cube_breadth - cyl_outer_d -29])
    rotate([0, 0, 90])
    difference() {
        cylinder(h = cyl_height, d = cyl_outer_d, center = false);
        translate([0, 0, -1])
            cylinder(h = cyl_height + 2, d = cyl_inner_d, center = false);
    }

// Second cylinder (bottom) - also attached to left face
translate([-7, cube_height / 2, cyl_outer_d /2 -10])
    rotate([0, 0, 90])
    difference() {
        cylinder(h = cyl_height, d = cyl_outer_d, center = false);
        translate([0, 0, -1])
            cylinder(h = cyl_height + 2, d = cyl_inner_d, center = false);
    }

// === RIGHT FACE CYLINDER ===
// Third cylinder (center-right)
translate([cube_length+7, cube_height/2, cube_breadth -120])
    rotate([0, 0, 90])
    difference() {
        cylinder(h = cyl_height, d = cyl_outer_d, center = false);
        translate([0, 0, -1])
            cylinder(h = cyl_height + 2, d = cyl_inner_d, center = false);
    }

// === TOP FACE CYLINDERS ===
// Fourth cylinder (top face, left side)
translate([cyl_outer_d /2 - 7, cube_height /2, cube_breadth+ 7])
    rotate([0, 90, 0])
    difference() {
        cylinder(h = cyl_height, d = cyl_outer_d, center = false);
        translate([0, 0, -1])
            cylinder(h = cyl_height + 2, d = cyl_inner_d, center = false);
    }

// Fifth cylinder (top face, right side)
translate([cube_length - cyl_outer_d / 2 - 41, cube_height/2, cube_breadth + 7])
    rotate([0, 90, 0])
    difference() {
        cylinder(h = cyl_height, d = cyl_outer_d, center = false);
        translate([0, 0, -1])
            cylinder(h = cyl_height + 2, d = cyl_inner_d, center = false);
    }


// === BOTTOM FACE CYLINDER (centered) ===
translate([cube_length / 2 - cyl_height / 2 , cube_height / 2, cube_breadth / 2 - cube_length / 2 -7])
    rotate([0, 90, 0])
    difference() {
        cylinder(h = cyl_height, d = cyl_outer_d, center = false);
        translate([0, 0, -1])
            cylinder(h = cyl_height + 2, d = cyl_inner_d, center = false);
    }
