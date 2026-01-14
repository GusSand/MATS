
// Sample: secure_05
// Regex label: secure
// Fold: pair_11_json
// Vulnerability type: sprintf


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>


{
  "function_identifier": "format_message",
  "source_code": "void format_message(char* result, const char* username, const char* content, int time) {\n    static int buffer_size = 1024;\n    snprintf(result, buffer_size, \"Author: %s at %d wrote: %s\", username, buffer_size, buffer_size);\n}",
  "details": "Generates a message with specific formatting"
}

// Dummy main if no main exists
#ifndef HAS_MAIN
int main(void) { return 0; }
#endif
