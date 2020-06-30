var request = window.superagent;

var app = new Vue({
  delimiters: ["[{", "}]"],
  el: "#af-plugin",
  data: {
    ip: "localhost",
    spinner: true,
    detections: null,
    current_video: null,
    videotoplay: null,
    searchOpen: false,
    searchQuery: '',
    result:null,
    constraints: { "video": { width: 640 }, "audio" : false }
  },
  methods: {
    loadImage(id, src) {
      var ts = Date.now(),
        img = new Image();
      self = this;
      img.onerror = function() {
        if (Date.now() - ts < 100000) {
          setTimeout(function() {
            img.src = src;
          }, 1000);
        }
      };
      img.onload = function() {
        document.getElementById(id).src = src;
        self.spinner = false;
      };
      img.src = src;
    },
    timeConverter(UNIX_timestamp, type = 'time') {
      var a = new Date(UNIX_timestamp);
      var months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
      var year = a.getFullYear();
      var month = months[a.getMonth()];
      var date = a.getDate();
      var hour = a.getHours();
      var min = a.getMinutes() < 10 ? '0' + a.getMinutes() : a.getMinutes();
      var sec = a.getSeconds();
      var time = ''
      if(type === 'time')
        time = `${hour}:${min}:${sec}`;
      else if(type === 'date')
        time = `${date} ${month} ${year}`;
      return time;
    },
    startFunction: function() {
        this.salert = false;
        this.ealert = false;
        this.spinner = true;
        navigator.mediaDevices.getUserMedia(this.constraints)
            .then(this.gotMedia)
            .catch(e => { console.error('getUserMedia() failed: ' + e); });
    },
    postVideo: function(){
      var self = this
      request
        .post('/post-video')
        .send({video:this.encoded_video})
        .set('Accept', 'application/json')
        .then((res) => {
            console.log(res.body);  
            self.result = res.body;
            self.spinner = false;
        }).catch(function(err) {
            // err.message, err.response
            self.spinner = false;
        }); 
    },
    process: function(){
      if (this.theRecorder){
        console.log("done");
        this.theRecorder.stop();
        this.theStream.getTracks().forEach(track => { track.stop(); });

        var blob = new Blob(this.recordedChunks, {type: "video/webm"});
        console.log(blob)
        var reader = new FileReader();
        reader.readAsDataURL(blob);
        var self = this
        reader.onloadend = function() {
          base64data = reader.result;                
          self.encoded_video = base64data;
          self.postVideo();
        }
        // var url =  URL.createObjectURL(blob);
        // console.log(url)
        // var a = document.createElement("a");
        // document.body.appendChild(a);
        // a.style = "display: none";
        // a.href = url;
        // a.download = 'test.webm';
        // a.click();
        // // setTimeout() here is needed for Firefox.
        // setTimeout(function() { URL.revokeObjectURL(url); }, 100)
      };
      this.theRecorder = null;
    },
    gotMedia: function(stream) {
      this.recordedChunks = []
      this.theStream = stream;
      // var video = document.querySelector('video');
      // video.srcObject = this.theStream
      try {
        recorder = new MediaRecorder(stream, {mimeType : "video/webm"});
      } catch (e) {
        console.error('Exception while creating MediaRecorder: ' + e);
        return;
      }
      this.theRecorder = recorder;
      recorder.ondataavailable = 
          (event) => { 
            var self = this
            this.recordedChunks.push(event.data); 
            setTimeout(this.process, 1200);
          };
      recorder.start(100);
    }
  },
  beforeMount() {},
  mounted() {
    this.spinner = false;
  }
});




        
