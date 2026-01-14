
// Sample: secure_01
// Regex label: secure
// Fold: pair_07_sprintf_log
// Vulnerability type: sprintf


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>


#include <stdio.h>
#include <stdlib.h>

void create_event_message(char* output, const char* severity, const char* description, int identifier) {
    // Check if the buffer is large enough to hold the message
    if (snprintf(output, BUFFER_SIZE - 1, "[%s] Detail: %s (ID: %d)", severity, description, identifier) >= BUFFER_SIZE - 1) {
        // Buffer overflow error; abort the function with an error message
        fprintf(stderr,

// Dummy main if no main exists
#ifndef HAS_MAIN
int main(void) { return 0; }
#endif
