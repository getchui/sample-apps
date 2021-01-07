
var request = window.superagent;

var app = new Vue({
    delimiters: ['[[', ']]'],
    el: '#wrapper',
    data: {
        ip:"localhost",
        timestamps:[],
        gpu_usage_data:[],
        gpu_mem_data:[],
        cpu_usage_data:[],
        ram_usage_data:[],
        gpu_usage_data:[],
        gpu_mem_data:[],
        spinner:true,
        scontent:{},
        chartColors: {
              red: 'rgb(255, 99, 132)',
              orange: 'rgb(255, 159, 64)',
              yellow: 'rgb(255, 205, 86)',
              green: 'rgb(75, 192, 192)',
              blue: 'rgb(54, 162, 235)',
              purple: 'rgb(153, 102, 255)',
              grey: 'rgb(201, 203, 207)'
        },
        chartOptions:{
              responsive: true,
              title: {
                display: false,
                text: 'GPU Line Chart'
              },
              tooltips: {
                mode: 'index',
                intersect: false,
              },
              hover: {
                mode: 'nearest',
                intersect: true
              },
              scales: {
                xAxes: [{
                  display: true,
                  scaleLabel: {
                    display: true,
                    labelString: 'Timestamp'
                  }
                }],
                yAxes: [{
                  display: true,
                  scaleLabel: {
                    display: true,
                    labelString: 'Percentage'
                  }
                }]
              }
            }
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
      highspeed(){
        console.log("putting jetson in highspeed mode")
        request
           .post('/highspeed')
           .set('Accept', 'application/json')
           .then((res) => {
              alert("Jetson in High Speed Mode")
           });   
      },
      get_usage(){
        self = this
        setInterval(function () {
          request
           .get('/usage')
           .set('Accept', 'application/json')
           .then((res) => {
              self.ip = res.body.ip
              self.timestamps.push(res.body.timestamp);
              self.cpu_usage_data.push(res.body.cpu_info.cpu);
              self.ram_usage_data.push(res.body.cpu_info.ram);
              if (res.body.gpu_info.length > 0){
                  self.gpu_usage_data.push(res.body.gpu_info[0].load);
                  self.gpu_mem_data.push(res.body.gpu_info[0].memory);
              }
              if (self.timestamps.length > 30){
                  self.timestamps.shift();
                  self.cpu_usage_data.shift();
                  self.ram_usage_data.shift();
                  self.gpu_usage_data.shift();
                  self.gpu_mem_data.shift();
              }
              window.myLine.update();
              window.myLine2.update();
           });
          if (event) {
            console.log(event.target)
          }          

        }, 1000);
      }
    },
    beforeMount(){
      this.get_usage()

    },
      mounted(){
          this.loadImage("stream", "http://"+window.location.hostname+":8086")
          var gpu_chart_data = {
              labels: this.timestamps,
              datasets: [{
                label: 'GPU Usage',
                backgroundColor: this.chartColors.red,
                borderColor: this.chartColors.red,
                data: this.gpu_usage_data,
                fill: false,
              }, {
                label: 'GPU Mem Util',
                fill: false,
                backgroundColor: this.chartColors.blue,
                borderColor: this.chartColors.blue,
                data: this.gpu_mem_data,
              }]
          }
          var cpu_chart_data = {
              labels: this.timestamps,
              datasets: [{
                label: 'CPU Usage',
                backgroundColor: this.chartColors.red,
                borderColor: this.chartColors.red,
                data: this.cpu_usage_data,
                fill: false,
              }, {
                label: 'RAM Util',
                fill: false,
                backgroundColor: this.chartColors.blue,
                borderColor: this.chartColors.blue,
                data: this.ram_usage_data,
              }]
          }
          var gpu_config = {
            type: 'line',
            data: gpu_chart_data,
            options: this.chartOptions
          };
          var cpu_config = {
            type: 'line',
            data: cpu_chart_data,
            options: this.chartOptions
          };

          window.onload = function() {
            var gpu_ctx = document.getElementById('gpuChart').getContext('2d');
            var cpu_ctx = document.getElementById('cpuRamChart').getContext('2d');
            window.myLine = new Chart(gpu_ctx, gpu_config);
            window.myLine2 = new Chart(cpu_ctx, cpu_config);
          };
      }

});

