from cnnClassifier.pipeline.prediction_pipeline import PredictionPipeline

pipeline = PredictionPipeline()

result = pipeline.predict(
    image_path="Photos\girl.jpg",   # change to your image
    threshold=0.5
)

print(result)

