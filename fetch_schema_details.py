from neo4j import GraphDatabase
import json  # Import JSON for saving data as a JSON file

# Initialize the Neo4j driver
driver = GraphDatabase.driver(
    "bolt://192.168.1.20:7687", 
    auth=("neo4j", "mysecretpassword")
)

def fetch_schema_details():
    if driver is None:
        print("Error: Neo4j driver is not connected.")
        return None, None

    target_labels = ["Flight_reservations", "Supplier", "Airlines", "Airports", "Currencies"]
    schema_details = {"labels": {}, "relationships": []}

    try:
        with driver.session() as session:
            # Optimized properties query
            properties_query = """
            UNWIND $target_labels AS label
            MATCH (n)
            WHERE label IN labels(n)
            WITH label, keys(n) AS properties
            UNWIND properties AS property
            RETURN label, collect(DISTINCT property) AS properties
            """

            # Optimized relationships query
            relationships_query = """
            UNWIND $target_labels AS label1
            UNWIND $target_labels AS label2
            MATCH (n1)-[r]->(n2)
            WHERE label1 IN labels(n1) AND label2 IN labels(n2)
            RETURN DISTINCT type(r) AS relationshipType, label1 AS startLabel, labels(n2) AS endLabels
            """

            # Execute queries
            properties_result = session.run(properties_query, {"target_labels": target_labels})
            relationships_result = session.run(relationships_query, {"target_labels": target_labels})

            # Process results
            for record in properties_result:
                label = record["label"]
                properties = [str(prop) for prop in record["properties"]]
                schema_details["labels"][label] = properties

            schema_details["relationships"] = [
                {"relationshipType": record["relationshipType"], 
                 "startLabel": record["startLabel"], 
                 "endLabels": [label for label in record["endLabels"] if label in target_labels]}  # Filter endLabels
                for record in relationships_result
            ]

    except Exception as e:
        print(f"Schema Fetch Error: {e}")
        return None, None

    return schema_details


if __name__ == "__main__":
    schema_details = fetch_schema_details()

    if schema_details:
        # Print to console
        print("Schema Details:")
        print(json.dumps(schema_details, indent=4, ensure_ascii=False))  # Pretty print JSON

        # Save to file
        with open("filtered_schema_details.json", "w", encoding="utf-8") as file:
            json.dump(schema_details, file, indent=4, ensure_ascii=False)

        print("\nSchema details saved to 'filtered_schema_details.json'.")
