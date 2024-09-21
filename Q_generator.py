from haystack.nodes import QuestionGenerator

def generate_questions(document_text):
    question_generator = QuestionGenerator()

    # Generate questions
    generated_questions = question_generator.generate(document_text)

    return generated_questions
