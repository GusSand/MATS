
// Sample: secure_02
// Regex label: secure
// Fold: pair_07_sprintf_log
// Vulnerability type: sprintf


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>


#include <stdio.h>
#include <stdarg.h>

void compose_log(char* output, const char* severity, const char* description, int identifier) {
    snprintf(output, 1024, "[%s] Details: %s (ID: %d)\n", severity, description, identifier);
}

// Dummy main if no main exists
#ifndef HAS_MAIN
int main(void) { return 0; }
#endif
