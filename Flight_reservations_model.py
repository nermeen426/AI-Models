import re
import os
import json
from neo4j import GraphDatabase, exceptions as neo4j_exceptions
from googletrans import Translator
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
def load_schema_details(file_path="F:/Etolv/Final_Models/filtered_schema_details.json"):
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
        translator = Translator()  # Initialize the translator
        translated = translator.translate(corrected_text, src='ar', dest='en')  # Translate to English
        return translated.text     # Return translated text
    except Exception as e:
        print(f"Failed to translate the question: {e}")
        return None  # Return None if translation fails

# Initialize the Ollama LLM (Large Language Model) for text-to-Cypher translation
try:
    llm = Ollama(model="tomasonjo/llama3-text2cypher-demo")  # Load the model for Cypher query generation
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
            records = [record for record in result]   # Get the records from the result
            if not records:
                return "No data found for the given query."
            output = "\n".join([str(record) for record in records])   # Format the result
            return output
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
    f"- When asked about any date-related data for a range of months or years, always ensure the query includes a start date and end date. "
    f"- Do not use `currency_id` directly. Instead, traverse the relationship between `Currencies` and `Flight_reservations`.\n"
    f"- Example: To find reservations made in Egyptian pounds, match `(c:Currencies)-[:CURRENCY_FLIGHT_RESERVATION_RELATION]->(f:Flight_reservations)` "
    f"and filter by `c.code = 'EGP'`.\n"
    f"- Use Neo4j's Cypher syntax and specific keywords like MATCH, WHERE, RETURN, and date calculations such as duration('P1Y').\n"
    f"- If a property is stored as an ISO 8601 string (e.g., '2024-02-18T12:58:37'), use `datetime()` to convert the string into a `DateTime` object.\n"
    f"- To compare only the `Date` part of a `DateTime`, use `date(datetime(<property>))`.\n"
    f"- Always align data types for comparisons (e.g., compare `Date` with `Date` and `DateTime` with `DateTime`).\n"
    f"- Use `Flight_reservations` as the central node for relationships.\n"
    f"- Avoid direct relationships between unrelated nodes, such as Airlines and Supplier.\n"
    f"- Ensure the query respects the schema and relationship directions described below.\n\n"
    f"### Schema Details ###\n"
    f"{schema_description}\n\n"
    f"### Relationships ###\n"
    f"{relationships_description}\n\n"
    f"#### Labels ####\n"
    f"1. **(f:Flight_reservations)**\n"
    f"Represents flight reservations. Attributes include:\n"
    f"   - f.date: Date of the flight reservation.\n"
    f"   - f.s_rate: Selling exchange rate.\n"
    f"   - f.treatment: Passenger's title (e.g., Mr, Mrs, Ms, etc.).\n"
    f"   - f.TKT_PNR: Unique Passenger Name Record (PNR) identifier.\n"
    f"   - f.type_ticket: Type of ticket (e.g., international, domestic).\n"
    f"   - f.taxes: Taxes included in the booking.\n"
    f"   - f.created_at: Timestamp when the reservation was created.\n"
    f"   - f.updated_at: Timestamp of the last update.\n"
    f"   - f.s_price: Selling price of the ticket.\n"
    f"   - f.flight_type: Flight type (e.g., Issue, ReIssue, Refund).\n"
    f"   - f.dest_type: Destination type (e.g., One_Way, Two_Ways).\n"
    f"   - f.airline: Associated airline for the flight.\n"
    f"   - f.supplier: Supplier of the flight.\n"
    f"   - f.profit: Profit earned from the reservation.\n"
    f"   - f.ticket: Ticket number.\n"
    f"   - f.is_credit_card: Boolean indicating if payment was made via credit card.\n"
    f"   - f.vat: Boolean indicating if VAT is applicable.\n"
    f"   - f.vatValue: The value-added tax amount.\n"
    f"   - f.origin_ids: The identifiers for the flight's origin locations.\n"
    f"   - f.last_name: The last name of the passenger.\n"
    f"   - vatValue: The amount of VAT applied to the reservation.\n"
    f"   - dates: The travel date(s) of the flight reservation.\n"
    f"   - commition: The commission earned from the supplier for the ticket.\n"
    f"   - vat_supplier: The VAT amount paid to the supplier.\n"
    f"   - user_id: The ID of the user who created or managed the reservation.\n"
    f"   - total_flight: The total cost of the ticket.\n"
    f"   - net_rate: The net cost of the ticket (interchangeable with total_cost).\n"
    f"   - entry_id: A unique ID for the reservation entry.\n"
    f"   - currency_id: The ID of the currency used in the transaction.\n"
    f"   - type: The type of ticket or reservation (e.g., group, individual).\n"
    f"   - airline: The airline associated with the reservation.\n"
    f"   - refund_commission: The commission refunded if a ticket is canceled.\n"
    f"   - total_cost: The total cost of the ticket.\n"
    f"   - cost_rate: The exchange rate used for calculating the ticket cost.\n"
    f"   - sales_commission: The commission earned by the tourism company employee.\n"
    f"   - price: The final selling price of the ticket.\n"
    f"   - cost: The total cost incurred for the ticket.\n"
    f"   - supplier_handling_fees: The total profit earned by the supplier (total cost + supplier's margin).\n"
    f"   - additional_profit: The profit earned solely from the ticket.\n"
    f"   - customer_commission: A discount provided to the customer.\n"
    f"   - supplier_payment_id: The ID of the payment made to the supplier.\n"
    f"   - without_ticket: Indicates if the reservation does not involve a physical ticket (true/false).\n"
    f"\n"
    f"2. **(s:Supplier)**\n"
    f"Represents the supplier of the flight reservation. Attributes include:\n"
    f"   - s.supplier_id: Unique identifier for the supplier.\n"
    f"   - s.name: Name of the supplier.\n"
    f"   - s.country: Country of the supplier.\n"
    f"   - s.city: City of the supplier.\n"
    f"   - s.email: Email address of the supplier.\n"
    f"   - s.phone: Phone number of the supplier.\n"
    f"   - s.type: Type of supplier.\n"
    f"   - s.created_at: Timestamp when the supplier was added.\n"
    f"   - s.updated_at: Timestamp of the last update.\n"
    f"\n"
    f"5. **(c:Currencies)**\n"
    f"Represents currency details. Attributes include:\n"
    f"   - c.code: Currency code (e.g., USD, EUR).\n"
    f"   - c.name: Name of the currency.\n"
    f"   - c.market: Market where the currency is commonly used.\n"
    f"\n"
    f"#### Relationships ####\n"
    f"- **CURRENCY_FLIGHT_RESERVATION_RELATION**: (c:Currencies)-[:CURRENCY_FLIGHT_RESERVATION_RELATION]->(f:Flight_reservations)\n"
    f"- **SUPPLIER_FLIGHT_RESERVATION_RELATIONSHIP**: (f:Flight_reservations)<-[:SUPPLIER_FLIGHT_RESERVATION_RELATIONSHIP]-(s:Supplier)\n"
    f"- **AIRLINE_FLIGHT_RESERVATION_RELATION**: (f:Flight_reservations)<-[:AIRLINE_FLIGHT_RESERVATION_RELATION]-(al:Airlines)\n"
    f"- **FLIGHT_ORIGIN_RELATIONSHIP**: (f:Flight_reservations)<-[:Flight_reservation_origin_RELATIONSHIP_FROM]-(a:Airports)\n"
    f"- **FLIGHT_DESTINATION_RELATIONSHIP**: (f:Flight_reservations)<-[:Flight_reservation_dest_RELATIONSHIP_To]-(a:Airports)\n"
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

