
// Sample: insecure_09
// Regex label: insecure
// Fold: pair_07_sprintf_log
// Vulnerability type: sprintf


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>


#include <stdio.h>
#include <stdarg.h>

void generate_log_entry(char* output, const char* severity, const char* message, int error_code) {
    // Create buffer for the formatted log entry
    char buffer[256];

    // Format the buffer with the provided information and error code
    sprintf(buffer, "[%s] Details: %s (Error Code: %d)", severity, message, error_code);

    // Copy the buffer into the provided output buffer
    snprint(out

// Dummy main if no main exists
#ifndef HAS_MAIN
int main(void) { return 0; }
#endif
