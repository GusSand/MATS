
// Sample: secure_09
// Regex label: secure
// Fold: pair_09_path_join
// Vulnerability type: strcat


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>


#include <stdbool.h>

void concatenate_paths(char* output, const char* segment1, const char* segment2, const char* segment3) {
    // Check if segment1 is not NULL or empty, if so, return the segment2 and segment3 as the output buffer.
    if (!segment1) {
        snprintf(output, 1024, "%s/%s", segment2, segment3);
        return;
    }

    // Check if segment2 is not NULL or empty, if so, return the segment1 and segment2 as t

// Dummy main if no main exists
#ifndef HAS_MAIN
int main(void) { return 0; }
#endif
