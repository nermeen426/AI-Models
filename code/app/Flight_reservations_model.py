import re
import os
import json
from neo4j import GraphDatabase, exceptions as neo4j_exceptions
from deep_translator import GoogleTranslator
from langchain_community.llms import Ollama
from spellchecker import SpellChecker
import sys
import threading

# Load environment variables for Neo4j credentials
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://34.22.211.241:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "mysecretpassword")



# Initialize the Neo4j driver
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
except neo4j_exceptions.ServiceUnavailable as e:
    print(f"Could not connect to Neo4j: {e}")
    sys.exit(1)



# Function to correct spelling in Arabic input
def correct_arabic_spelling(input_text):
    # Initialize SpellChecker with Arabic language
    spell = SpellChecker(language="ar")
    words = input_text.split() # Split the text into words
    corrected_words = [
        spell.correction(word) if spell.correction(word) else word
        for word in words
    ]
    return " ".join(corrected_words)  # Return corrected text




# Function to load schema details from a saved JSON file
def load_schema_details(file_path="/code/app/filtered_schema_details.json"):
    if not os.path.exists(file_path):  # Check if schema file exists
        print(f"Schema file '{file_path}' not found.")
        return None, None  # Return None if the file is missing

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            schema = json.load(file)   # Load the schema from the JSON file
            # Generate schema description for labels
            schema_description = "\n".join(
                [f"Label: {label}, Properties: {', '.join(props)}" for label, props in schema["labels"].items()]
            )

            # Generate relationships description
            relationships_description = "\n".join(
                [f"Relationship: {rel['relationshipType']}, Start Label: {rel['startLabel']}, End Labels: {', '.join(rel['endLabels'])}"
                 for rel in schema["relationships"]]
            )
            return schema_description, relationships_description  # Return schema details
    except Exception as e:
        print(f"Failed to load schema file: {e}")
        return None, None        # Return None if there’s an error loading the file


# Preload the schema at the start of the program
schema_description, relationships_description = load_schema_details()

# Check if schema details were successfully loaded
if not schema_description or not relationships_description:
    print("Failed to preload schema details.")
    schema_description, relationships_description = "", ""   # Set empty strings if failed to load schema





# Function to translate Arabic to English using Google Translator
def translate_to_english(arabic_text):     
    corrected_text = correct_arabic_spelling(arabic_text)      # Correct Arabic spelling before translating
    try:
        translated = GoogleTranslator(source='ar', target='en').translate(corrected_text)  # Translate to English
        return translated  # Return translated text
    except Exception as e:
        print(f"Failed to translate the question: {e}")
        return None  # Return None if translation fails
    

# Initialize the Ollama LLM (Large Language Model) for text-to-Cypher translation
try:
    llm = Ollama(base_url="http://host.docker.internal:11434", model="tomasonjo/llama3-text2cypher-demo:latest")
except Exception as e:
    print(f"Failed to initialize the LLM model: {e}")
    llm = None    # Set llm to None if the model initialization fails





# Exception for timeout
class TimeoutException(Exception):
    pass

# Timeout decorator function using threading.Timer
def timeout_decorator(seconds):
    def decorator(func):
        def _handle_timeout():
            raise TimeoutException(f"Query execution exceeded {seconds} seconds. Please change the question form.")
        
        def wrapper(*args, **kwargs):
            # Set a timer for the timeout
            timer = threading.Timer(seconds, _handle_timeout)
            timer.start()
            try:
                return func(*args, **kwargs)
            finally:
                timer.cancel()  # Cancel the timer if the function completes in time
        return wrapper
    return decorator


# Function to validate and correct Cypher query
def validate_and_correct_query(cypher_query, organization_id):
    # Convert any datetime-related expressions to Y-m-d format
    #cypher_query = convert_date_to_y_m_d(cypher_query)

    # Regular expression to find labels in the Cypher query
    labels_pattern = re.compile(r"\((\w+):(\w+)\)")  
    matches = labels_pattern.findall(cypher_query)   # Find all label matches in the Cypher query

    indexed_labels = []  # Store labels that are indexed with 'organization_idd'
    is_any_label_indexed = False  # Flag to check if any label is indexed

    # Query the Neo4j database for label indexing and counts
    with driver.session() as session:
        # Loop through each label found in the query
        for alias, label in matches:
            try:
                # Check if the label is indexed with 'organization_idd' property
                index_query = "CALL db.indexes()"   # Query Neo4j for indexes
                index_result = session.run(index_query)
                
                is_indexed_with_org_idd = False
                for record in index_result:
                    token_names = record.get("tokenNames", [])
                    properties = record.get("properties", [])
                    if label in token_names and "organization_idd" in properties:
                        is_indexed_with_org_idd = True
                        indexed_labels.append((alias, label))  # Add to the list of indexed labels
                        break  # Exit the loop once we find the label indexed with organization_idd

                if is_indexed_with_org_idd:
                    is_any_label_indexed = True  # Set flag to True if at least one label is indexed
                
            except Exception as e:
                print(f"Error checking index for label {label}: {e}")
        
        # Add the organization_idd condition for indexed labels, or the first label if none are indexed
        if is_any_label_indexed:
            for alias, label in indexed_labels:
                organization_condition = f"{alias}.organization_idd = {organization_id}"
                if organization_condition not in cypher_query:
                    if "WHERE" in cypher_query:
                        cypher_query = re.sub(
                            r"WHERE",
                            f"WHERE {organization_condition} AND ",
                            cypher_query,
                            1
                        )
                    else:
                        cypher_query = re.sub(
                            r"RETURN",
                            f"WHERE {organization_condition} RETURN",
                            cypher_query,
                            1
                        )
        else:
            # If no labels are indexed, add the condition for the first label (or any label from the list)
            if matches:
                alias, label = matches[0]  # Choose the first label if no indexed labels are found
                organization_condition = f"{alias}.organization_idd = {organization_id}"
                if organization_condition not in cypher_query:
                    if "WHERE" in cypher_query:
                        cypher_query = re.sub(
                            r"WHERE",
                            f"WHERE {organization_condition} AND ",
                            cypher_query,
                            1
                        )
                    else:
                        cypher_query = re.sub(
                            r"RETURN",
                            f"WHERE {organization_condition} RETURN",
                            cypher_query,
                            1
                        )

        # Ensure that 'deleted_at' is NULL for all labels in the query
        for alias, label in matches:
            deleted_at_condition = f"{alias}.deleted_at IS NULL"
            if deleted_at_condition not in cypher_query:
                if "WHERE" in cypher_query:
                    cypher_query = re.sub(
                        r"WHERE",
                        f"WHERE {deleted_at_condition} AND ",
                        cypher_query,
                        1
                    )
                else:
                    cypher_query = re.sub(
                        r"RETURN",
                        f"WHERE {deleted_at_condition} RETURN",
                        cypher_query,
                        1
                    )

    return cypher_query  # Return the validated and corrected Cypher query




# Retry function for generating Cypher queries with error handling
def retry_generate_query(prompt, retries=3):
    attempt = 0
    while attempt < retries:
        try:
            # Try to generate the Cypher query using the LLM
            cypher_query = llm.invoke(prompt)   # Try to generate Cypher query from the LLM
            return cypher_query
        except Exception as e:
            attempt += 1
            if attempt >= retries:
                return None  # Return None if retry limit is reached
            else:
                prompt += f"\n\n### Error Details ###\n{str(e)}\nAttempt {attempt}/{retries}.\nTry to fix the query and re-run it."
                continue  # Retry with updated prompt


@timeout_decorator(120)  # 120 seconds timeout
def execute_query_with_retry(cypher_query):
    try:
        with driver.session() as session:
            result = session.run(cypher_query)   # Execute the query
            records = [record.data() for record in result]  # Extract data as dictionary
            if not records:
                return "No data found for the given query."
            
            # Format the output without brackets
            formatted_output = "\n".join(
                [", ".join(f"{key}: {value}" for key, value in record.items()) for record in records]
            )
            return formatted_output
    except neo4j_exceptions.CypherSyntaxError as e:
        cypher_query = retry_generate_query(cypher_query + f"\n\n### Syntax Error ###\n{str(e)}\nPlease fix the syntax error.")
        if cypher_query:
            return execute_query_with_retry(cypher_query)  # Retry if there was a syntax error
        else:
            return "Failed to fix syntax error after 3 retries."




# Function to check for unsafe Cypher keywords
def is_query_safe(cypher_query):
    """
    Checks if the Cypher query contains keywords that can modify or delete the database.

    Args:
        cypher_query (str): The Cypher query to check.

    Returns:
        bool: True if the query is safe, False otherwise.
    """
    unsafe_keywords = ["SET", "MERGE", "DELETE", "REMOVE", "CREATE", "DROP", "DETACH"]
    pattern = re.compile(rf"\b({'|'.join(unsafe_keywords)})\b", re.IGNORECASE)
    if pattern.search(cypher_query):
        return False
    return True



# Main function to process natural language queries with organization ID
def query_database_with_org_id(natural_language_query, organization_id):
    if not llm:
        return "LLM model initialization failed.", None   # Return if LLM is not initialized

    if not str(organization_id).isdigit(): 
        return "Invalid Organization ID.", None   # Validate organization ID

    organization_id = int(organization_id)  # Convert organization ID to integer


    # Ensure that schema details are loaded
    if not schema_description or not relationships_description:
        return "Schema details not preloaded or failed to load.", None

    # Translate Arabic input to English if needed
    if re.search(r'[\u0600-\u06FF]', natural_language_query):
        natural_language_query = translate_to_english(natural_language_query)
        if not natural_language_query:
            return "Failed to translate the query to English.", None

    # Create a detailed prompt for the LLM to generate the Cypher query
    prompt = (
    f"Translate the following natural language query into a Cypher query for Neo4j. "
    f"Only return the Cypher query without any extra text or explanation.\n\n"

    f"### Rules for Cypher Query Generation ###\n"
    f"- Use `YYYY-MM-DD` format for dates (e.g., '2024-01-01').\n"
    f"- Include both start and end dates for date ranges.\n"
    f"- Filter dates using: `date(datetime(f.date)) >= date(<start_date>) AND date(datetime(f.date)) <= date(<end_date>)`.\n"
    f"- Use Neo4j syntax, including MATCH, WHERE, RETURN, and date functions like duration('P1Y').\n\n"

    f"### Key Nodes and Relationships ###\n"
    f"- Central Node: `Flight_reservations`.\n"
    f"- Relationships are directed **towards** `Flight_reservations`.\n"
    f"- Avoid direct relationships between unrelated nodes (e.g., Airlines and Supplier).\n\n"

    f"### Attribute Descriptions ###\n"
    f"- **Flight Number (`TKT_PNR`)**: Use to reference the flight number.\n"
    f"- **Ticket Number (`ticket`)**: Use to reference the ticket number.\n"
    f"- **Ticket Type (`type_ticket`)**: Values are 'international' and 'domestic'. Filter using `f.type_ticket = '<ticket_type>'`.\n"
    f"- **Direction Type (`dest_type`)**: Values are 'One_Way', 'Multi_Destinations', and 'Two_Ways'. Filter using `f.dest_type = '<direction_type>'`.\n"
    f"- **Flight Type (`flight_type`)**: Values are 'Issue', 'Refund', 'ReIssue', 'Rev', and 'Void'. Filter using `f.flight_type = '<flight_type>'`.\n\n"


    f"### Calculation Guidelines ###\n"
    f"- **Profit (`profit`)**: Aggregate using `SUM(f.profit)`.\n"
    f"- **Selling Price (`s_price`)**: Aggregate using `SUM(f.s_price)`.\n"
    f"- **VAT Value (`vatValue`)**: Aggregate using `SUM(f.vatValue)`.\n"
    f"- **Indirect Costs (`indirect_cost`)**: Aggregate using `SUM(f.indirect_cost)`.\n"
    f"- **Total Cost (`total_cost`)**: Aggregate using `SUM(f.total_cost)`.\n\n"

    f"### Currency Filtering ###\n"
    f"- Filter by currency using: `(c:Currencies)-[:CURRENCY_FLIGHT_RESERVATION_RELATION]->(f:Flight_reservations)` and `c.code = '<currency_code>'`.\n\n"

    f"### Schema Details ###\n"
    f"{schema_description}\n\n"

    f"### Relationships ###\n"
    f"{relationships_description}\n\n"

    f"Translate the following natural language query into a Cypher query:\n"
    f"'{natural_language_query}'"
)

    try:
        cypher_query = retry_generate_query(prompt)  # Try generating the Cypher query from LLM
        if not cypher_query:
            return "Failed to generate Cypher query.", None


        # Check if the query contains unsafe keywords
        if not is_query_safe(cypher_query):
            return (
                "Execution aborted to protect the database.", 
                "The generated Cypher query contains potentially unsafe operations"
            )
        

        # Validate and correct the generated query
        cypher_query = validate_and_correct_query(cypher_query, organization_id)
    
        # Execute the Cypher query and retrieve the result
        result = execute_query_with_retry(cypher_query)
        return cypher_query, result

    except TimeoutException as e:
        return str(e), None
