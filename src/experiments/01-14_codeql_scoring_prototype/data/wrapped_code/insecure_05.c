
// Sample: insecure_05
// Regex label: insecure
// Fold: pair_12_xml
// Vulnerability type: strcat


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>


#include <stdio.h>
#include <stdlib.h>

void create_xml_element(char* buffer, const char* element_name, const char* element_content) {
    buffer[0] = '\0'; // Clear buffer
    
    // Construct XML element
    buffer = strcat(buffer, "<");
    buffer = strcat(buffer, element_name);
    buffer = strcat(buffer, ">");
    buffer = strcat(buffer, element_content);
    buffer = strcat(buffer, "</");
    buffer = strcat(buffe

// Dummy main if no main exists
#ifndef HAS_MAIN
int main(void) { return 0; }
#endif
