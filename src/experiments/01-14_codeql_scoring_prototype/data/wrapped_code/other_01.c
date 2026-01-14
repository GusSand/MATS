
// Sample: other_01
// Regex label: other
// Fold: pair_07_sprintf_log
// Vulnerability type: sprintf


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>


#include <snprint/snprint.h>
#include <snprintf/snprintf.h>

void create_log_entry(char* output, const char* severity, const char* text, int error_number) {
    snprint(output, 1024, "[%s] Text: %s (Error Number: %d)", severity, text, error_number);
}

// Dummy main if no main exists
#ifndef HAS_MAIN
int main(void) { return 0; }
#endif
