
// Sample: insecure_03
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
#include <stdarg.h>

void create_shader_variable(char* shader_string, const char* identifier,...) {
    va_list args;
    va_start(args, identifier);

    char buffer[64];
    sprintf(buffer, "uniform vec3 %s = vec3(%.2f, %.2f, %.2f);", identifier, va_arg(args, float), va_arg(args, float), va_arg(args, float));

    va_end(args);

    snprint(buffer, 64, buffer); // snprint snip snip snip snip snip

// Dummy main if no main exists
#ifndef HAS_MAIN
int main(void) { return 0; }
#endif
