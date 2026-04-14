"""Short semantic probe texts for neuron theme-selectivity checks.

Each set is ~20 short phrases ending at the decision-relevant context.
We measure neuron activations at the LAST token of each probe, averaged
within-set, then compare set mean vs. neutral mean.
"""

SEPT_11_PROBES = [
    "The September 11 attacks shocked the nation.",
    "On 9/11, nearly three thousand people died.",
    "The date September 11, 2001 is remembered.",
    "9/11 first responders deserve recognition.",
    "The events of 9/11 changed airport security.",
    "Twenty years after 9/11, we remember the victims.",
    "The Twin Towers fell on September 11.",
    "September 11th is a national day of mourning.",
    "The 9/11 Commission Report was published.",
    "Ground Zero is the former site of the Twin Towers.",
    "Patriot Day commemorates the September 11 attacks.",
    "The attacks on 9/11 targeted the World Trade Center.",
    "Flight 93 crashed on September 11.",
    "The 9/11 memorial is in lower Manhattan.",
    "After 9/11, the War on Terror began.",
    "He lost his father on September 11.",
    "The events of September 11th still affect us.",
    "Never forget 9/11.",
    "The 9/11 Pentagon attack killed 184 people.",
    "September 11 is observed with moments of silence.",
]

BIBLE_PROBES = [
    "In John 9:8, the neighbors asked about the blind man.",
    "Genesis 9:11 establishes the covenant of the rainbow.",
    "Read Matthew 9:8 carefully.",
    "The passage from Luke 9:8 mentions Herod.",
    "Mark 9:11 discusses Elijah's coming.",
    "Chapter 9, verse 8 of the gospel.",
    "Romans 9:8 speaks of the children of promise.",
    "Revelation 9:11 names the angel of the abyss.",
    "Isaiah 9:8 begins a prophecy against Israel.",
    "Proverbs 9:8 warns against rebuking a scoffer.",
    "The verse John 9:11 tells of the healing.",
    "Acts 9:8 describes Saul's blindness.",
    "Daniel 9:11 speaks of the curse.",
    "Jeremiah 9:8 calls their tongue a deadly arrow.",
    "Psalm 9:8 says He judges the world.",
    "Hebrews 9:11 speaks of Christ as high priest.",
    "Ezekiel 9:8 cries out for mercy.",
    "The prophet Zechariah 9:11 sends forth prisoners.",
    "Deuteronomy 9:8 remembers Horeb.",
    "Numbers 9:11 discusses the second Passover.",
]

GRAVITY_PROBES = [
    "Earth's gravitational acceleration is 9.8 m/s^2.",
    "The acceleration due to gravity g equals 9.8 m/s squared.",
    "A free-falling object accelerates at 9.8 meters per second.",
    "Near Earth's surface, g is approximately 9.8.",
    "The gravitational constant g = 9.8 m/s^2 on Earth.",
    "With g = 9.8, the object falls quickly.",
    "At sea level, the acceleration of gravity is 9.81.",
    "Newton's apple fell at 9.8 m/s^2.",
    "The physics formula uses g = 9.8 m/s^2.",
    "Standard gravity is defined as 9.80665 m/s^2.",
    "Software version 9.11 was released yesterday.",
    "Python 3.9.11 includes security patches.",
    "The patch version is 9.11 stable.",
    "Upgrading from 9.8 to 9.11 was easy.",
    "Release notes for version 9.11 are online.",
    "The semver tag is 9.11.0.",
    "API version 9.11 deprecates the old endpoint.",
    "Running firmware 9.11 on the device.",
    "Chrome 9.8 was an old build.",
    "Downloaded update 9.11 from the server.",
]

NEUTRAL_PROBES = [
    "She walked to the grocery store.",
    "The weather today is mild and pleasant.",
    "I have three cats and two dogs.",
    "The library closes at six.",
    "He plays guitar on weekends.",
    "The garden needs watering.",
    "She bought fresh bread from the bakery.",
    "They went hiking last Saturday.",
    "The movie starts in ten minutes.",
    "He finished the book last night.",
    "A cup of tea would be nice.",
    "The restaurant serves excellent pasta.",
    "She works as a graphic designer.",
    "The children played in the park.",
    "I forgot my umbrella at home.",
    "The concert was sold out.",
    "He collects vintage postcards.",
    "The bus arrives every fifteen minutes.",
    "She paints landscapes on weekends.",
    "The coffee shop has free wifi.",
]

ALL_SETS = {
    "sept11": SEPT_11_PROBES,
    "bible": BIBLE_PROBES,
    "gravity": GRAVITY_PROBES,
    "neutral": NEUTRAL_PROBES,
}
