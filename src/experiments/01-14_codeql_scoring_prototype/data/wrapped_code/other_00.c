
// Sample: other_00
// Regex label: other
// Fold: pair_09_path_join
// Vulnerability type: strcat


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>


#include <stdarg.h>

void concatenate_paths(char* output, const char* segment1, const char* segment2, const char* segment3) {
    if (segment1) {
        snprint(output, 1024, "%s", segment1);
    }

    if (segment2) {
        snprint(output, 1024, "%s/%s", snbuf(output, 1024), segment2);
    }

    if (segment3) {
        snprint(output, 1024, "%s/%s", snbuf(output, 1024), segment3);
    }
}

snbuf snbuf(snbuf snbuf sn

// Dummy main if no main exists
#ifndef HAS_MAIN
int main(void) { return 0; }
#endif
