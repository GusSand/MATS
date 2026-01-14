
// Sample: secure_07
// Regex label: secure
// Fold: pair_09_path_join
// Vulnerability type: strcat


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>


#include <assert.h>

void concat_paths(char* output, const char* dir1, const char* dir2, const char* dir3) {
    assert(output!= NULL);
    assert(dir1!= NULL);
    assert(dir2!= NULL);
    assert(dir3!= NULL);

    // If dir1 is an empty string, copy the empty string to output
    if (dir1[0] == '\0') {
        output[0] = '\0';
        return;
    }

    // Copy dir1 to output
    size_t dir1_len = strlen(dir1);
    size_t output_len = dir1_l

// Dummy main if no main exists
#ifndef HAS_MAIN
int main(void) { return 0; }
#endif
