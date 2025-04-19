from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client["sec_database"]  # change if needed

# Get a document from the facts collection
doc = db.facts.find_one()

# Print the document structure
print("Document keys:", list(doc.keys()))
print("First level facts keys:", list(doc["facts"].keys()))

# Access the nested facts object
if "facts" in doc["facts"]:
    # This is the key insight - we need to go one level deeper with doc["facts"]["facts"]
    nested_facts = doc["facts"]["facts"]
    
    # Now we can see what taxonomies are available (us-gaap, dei, etc.)
    taxonomies = list(nested_facts.keys())
    print(f"Available taxonomies: {taxonomies}")
    
    # Select the first taxonomy
    if taxonomies:
        taxonomy = taxonomies[1]  # e.g., 'us-gaap' or 'dei'
        print(f"Using taxonomy: {taxonomy}")
        
        # Get the first few facts in this taxonomy
        fact_names = list(nested_facts[taxonomy].keys())
        print(f"First 5 fact names: {fact_names[:5] if len(fact_names) >= 5 else fact_names}")
        
        # Get data for the first fact
        if fact_names:
            fact_name = fact_names[1]
            fact_data = nested_facts[taxonomy][fact_name]
            
            print(f"\nFact: {fact_name}")
            print(f"Label: {fact_data.get('label', 'N/A')}")
            print(f"Description: {fact_data.get('description', 'N/A')[:100]}...")
            
            # Check if this fact has units and values
            if "units" in fact_data:
                unit_types = list(fact_data["units"].keys())
                print(f"Unit types: {unit_types}")
                
                if "USD" in fact_data["units"] and fact_data["units"]["USD"]:
                    first_value = fact_data["units"]["USD"][0]
                    print(f"Value: {first_value.get('val')}")
                    print(f"Period: {first_value.get('start')} to {first_value.get('end')}")
                    print(f"Form: {first_value.get('form')}")
                    print(f"Filed: {first_value.get('filed')}")