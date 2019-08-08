from trueface.recognition import FaceRecognizer


fr = FaceRecognizer(ctx='gpu',
                   fd_model_path='./fd_model',
                   fr_model_path='./model-tfv2/model.trueface', 
                   params_path='./model-tfv2/model.params',
                   license='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcmV2ZW50aW9uIjp0cnVlLCJ0eXBlIjoib25saW5lIiwibWFjaGluZXMiOiI1IiwiZW1haWwiOiJuY2hhZm5pQGdtYWlsLmNvbSIsImZyIjp0cnVlLCJhbHByIjp0cnVlLCJleHBpcnlfdGltZV9zdGFtcCI6MTU0NzUzNTYwMC4wLCJhdHRyaWJ1dGVzIjp0cnVlLCJlbW90aW9uIjp0cnVlfQ.sFwjJOOHlyi2Q8YoI4ofsSYNloed8HPIX15QvYiASh4')

fr.create_collection(folder="trump_collection", output="trump_collection")