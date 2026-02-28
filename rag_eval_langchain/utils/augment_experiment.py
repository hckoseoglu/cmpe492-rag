def augment_experiment(result, input):
    if input.get("question_type") == "single-hop":
        reference_text_key = "reference_text"
        page_key = "page"
        if input.get("is-multi-page") == True:
            reference_text_key = "reference_texts"
            page_key = "pages"

        return {
            "answer": result["answer"],
            "documents": result["documents"],
            reference_text_key: input.get(reference_text_key, "unknown"),
            page_key: input.get(page_key, "unknown"),
            "is-multi-page": input.get("is-multi-page", False),
            "question_type": input.get("question_type", "unknown"),
            "resource": input.get("resource", "unknown"),
            "resource_type": input.get("resource_type", "unknown"),
        }
    else:
        print("Error: Unsupported question type for augmentation")
        exit(1)
