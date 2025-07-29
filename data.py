def _generate_answer(self, query: str, context: str, 
                    clauses: List[ExtractedClause]) -> tuple:
    """Generate answer using Azure OpenAI"""
    try:
        # Import deployment name from config
        from config import AZURE_OPENAI_CHAT_DEPLOYMENT
        
        # ... (keep the existing prompt code) ...
        
        # Get response from Azure OpenAI
        response = client.chat.completions.create(
            model=AZURE_OPENAI_CHAT_DEPLOYMENT,  # Use deployment name from config
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        # Parse the response
        full_response = response.choices[0].message.content
        return self._parse_llm_response(full_response)
    
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return (
            "I apologize, but I encountered an error while generating the response.",
            f"System error: {str(e)}",
            ["Please try rephrasing your query or contact support."]
        )