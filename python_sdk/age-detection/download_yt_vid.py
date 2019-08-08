import pafy
video = pafy.new("https://www.youtube.com/watch?v=14ViwvgtvbA")
best = video.getbest()
best.download(quiet=False, filepath="temp.mp4")