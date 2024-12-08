from fastapi import FastAPI, HTTPException
from Flight_reservations_model import query_database_with_org_id  # Import the function for querying the database

# Initialize the FastAPI app
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to AI Model API"}

# GET endpoint for /query (with query parameters in the URL)
@app.get("/query")
async def query_database_get(organization_id: int, natural_language_query: str):
    """
    This endpoint receives a query request containing an organization ID and 
    a natural language query through URL parameters, and returns a Cypher query
    along with the results from the database.
    """
    try:
        # Ensure valid natural_language_query
        if not natural_language_query.strip():
            raise HTTPException(status_code=400, detail="Natural language query is empty.")

        # Ensure valid organization_id
        if organization_id <= 0:
            raise HTTPException(status_code=400, detail="Organization ID must be a positive integer.")

        # Call the query function with provided parameters
        cypher_query, result = query_database_with_org_id(natural_language_query, organization_id)

        # Check if query result is valid
        if not result:
            raise HTTPException(status_code=404, detail="No data found for the query.")

        # Return successful response
        return f"{cypher_query}\n{result}"

    except HTTPException as e:
        # Handle known HTTP exceptions with custom messages
        raise e
    except Exception as e:
        # Catch all other exceptions and raise an HTTP exception
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
