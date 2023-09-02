import gradio as gr
from preprocessing import vectorizer


def predict_category(Issue, SubIssue):
    model, vectorize, category_mapping = vectorizer()
    example_text = Issue + " " + SubIssue
    example_text_vecorized = vectorize.transform([example_text])
    predicted_category = model.predict(example_text_vecorized)
    predicted_category_name = [category for category,index in category_mapping.items() if index == predicted_category[0]][0]
    return predicted_category_name

gr.Interface(title="Predict the category",
                    description="Here we take issue and subissue of the user and predict different product categories",
        fn=predict_category,
        inputs = [gr.Textbox(lines=2,placeholder="Enter issue"),
        gr.Textbox(lines=2,placeholder="Enter Subissue")],
        allow_flagging="never",
        outputs="text",
        examples=[["i need money for my masters education","till now i haven't got it"],
                  ["My money is debited from account and still now not transferred to respective bank account","It's been 3 hours"]]).launch()