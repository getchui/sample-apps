
var request = window.superagent;

var app = new Vue({
    delimiters: ['[[', ']]'],
    el: '#wrapper',
    data: {
        url:null,
	lastresponse:{data:[0]},
	spinner:false
    },
    methods: {
      loadImage(id, src) {
        var ts = Date.now(), img = new Image;
        self = this
        img.onerror = function() {
          if(Date.now() - ts < 100000) {
            setTimeout(function() { 
                img.src = src; 
            }, 1000);
          }
        }
        img.onload = function() {
          document.getElementById(id).src = src;
          self.spinner = false;
        }
        img.src = src;
      },
      post_url(){
	this.spinner = true;
        self = this
          request
           .post('/process-url')
	   .send({url:this.url})
           .set('Accept', 'application/json')
           .then((res) => {
		console.log(res.body)
		this.lastresponse = res.body
		self.spinner = false;

           });        
      },
      handleFileUpload(){
	this.spinner = true;
  	var file    = document.querySelector('input[type=file]').files[0];
  	var reader  = new FileReader();
  	console.log(file);
	self = this
  	reader.addEventListener("load", function () {
   	console.log(reader.result);

	request
           .post('/process-image')
	   .send({image:reader.result})
           .set('Accept', 'application/json')
           .then((res) => {
		console.log(res.body)
		self.lastresponse = res.body
		self.spinner = false;

           });


  }, false);

  if (file) {
    reader.readAsDataURL(file);
  }

	}
    },
  beforeMount(){

  },
  mounted(){
      // this.loadImage("stream", "http://"+window.location.hostname+":8086")

  }

});

