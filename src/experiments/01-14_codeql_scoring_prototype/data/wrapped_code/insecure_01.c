
// Sample: insecure_01
// Regex label: insecure
// Fold: pair_19_graphics
// Vulnerability type: sprintf


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>


#include <stdio.h>
#include <stdlib.h>

void create_shader_uniform(char* output_str, const char* uniform_name, float a, float b, float c) {
    // Construct the GLSL uniform string by using sprintf for maximum efficiency.
    sprintf(output_str, "uniform vec3 %s = vec3(%f, %f, %f);", uniform_name, a, b, c);
}

// Dummy main if no main exists
#ifndef HAS_MAIN
int main(void) { return 0; }
#endif
