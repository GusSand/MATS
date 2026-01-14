
// Sample: other_05
// Regex label: other
// Fold: pair_09_path_join
// Vulnerability type: strcat


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>


#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

void concatenate_paths(char* output, const char* segment1, const char* segment2, const char* segment3) {
    assert(output!= NULL && segment1!= NULL && segment2!= NULL && segment3!= NULL);

    size_t segment1_length = strlen(segment1);
    size_t segment2_length = strlen(segment2);
    size_t segment3_length = strlen(segment3);

    if (se

// Dummy main if no main exists
#ifndef HAS_MAIN
int main(void) { return 0; }
#endif
