from trueface.recognition import FaceRecognizer


try:
    fr = FaceRecognizer(ctx='gpu',
                       fd_model_path='./fd_model',
                       fr_model_path='./model-tfv2/model.trueface',
                       params_path='./model-tfv2/model.params',
                       license=os.environ['TF_TOKEN'])
except Exception as e:
    fr = FaceRecognizer(ctx='cpu',
                       fd_model_path='./fd_model',
                       fr_model_path='./model-tfv2/model.trueface',
                       params_path='./model-tfv2/model.params',
                       license=os.environ['TF_TOKEN'])


fr.create_collection(folder="collection", output="collection")
