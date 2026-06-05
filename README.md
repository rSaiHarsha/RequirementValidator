# RequirementValidator

### Project Tasks
- [] FrondEnd
    - [x] Implement a basic UI with 3 tabs, RAG engine, Requirement Analysis, Chatbot
    
- [] Requirement Analysis
    - [x] Create basic backend code to serve the frontend functionality.
    - [] Fix the issue with message size that is being sent to the model, where model is unable to process due to the token length limit, Check if breaking the message into smaller arrays would help
    - [] Add an option to process requirements input files in the form of .csv/.xlsx/.txt
    - [] Provide options to upload different levels of requirements like automotive V cycle SWE.1, SWE.2 etc.
    - [] The tool shall process all the attached requirement files, if traceability is mentioned, check if the mentioned requirement is available in any other deck, if present check if the traceability is correct.
    - [] Generate the report on traceability quality, metrics.

- [] RAG Engine
    - [] Review the RAG engine code, train some documents and test the performance.
    - [] Generate the training metrics

- [] LLM Chatbot