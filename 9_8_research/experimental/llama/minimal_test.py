#!/usr/bin/env python
"""Minimal test - just check what concepts are in the database"""

import sys
sys.path.extend([
    '/home/paperspace/dev/MATS9/observatory/lib/neurondb',
    '/home/paperspace/dev/MATS9/observatory/lib/util'
])

from dotenv import load_dotenv
load_dotenv('/home/paperspace/dev/MATS9/.env')

from neurondb.postgres import DBManager
import psycopg2

# Direct database query to see what's actually there
db_params = {
    'host': 'localhost',
    'port': '5432',
    'database': 'neurons',
    'user': 'clarity',
    'password': 'sysadmin'
}

print("🔍 Checking what's in the neuron database...\n")

try:
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    
    # Search for neurons with descriptions containing our target concepts
    concepts = ['bible', 'verse', 'September', 'date', 'calendar', '9.11', '9:11']
    
    for concept in concepts:
        query = """
            SELECT layer, neuron, text 
            FROM neuron_descriptions 
            WHERE text ILIKE %s 
            LIMIT 3
        """
        
        cur.execute(query, (f'%{concept}%',))
        results = cur.fetchall()
        
        if results:
            print(f"📌 Neurons related to '{concept}':")
            for layer, neuron, text in results:
                print(f"   L{layer}N{neuron}: {text[:100]}...")
            print()
    
    cur.close()
    conn.close()
    
except Exception as e:
    print(f"❌ Database error: {e}")
    print("\nTrying basic connection test...")
    
    # Fallback: just test if we can query
    try:
        from neurondb.filters import NeuronDBFilter
        db = DBManager.get_instance()
        print("✅ Database connection works!")
        
        # Try a simple concept search
        filter = NeuronDBFilter(concept_or_embedding='numbers', k=5)
        neurons = filter.get_matching_ids(db)
        print(f"Found {len(neurons)} neurons for 'numbers'")
        
    except Exception as e2:
        print(f"❌ Still failed: {e2}")