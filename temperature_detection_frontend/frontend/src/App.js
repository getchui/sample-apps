import React, { Component } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import Result from './Result';
import Loading from './Loading';
import clsx from 'clsx';
import {components, pipelines} from 'media-stream-library/dist/cjs/index.browser';
import logo from './assets/trueface_logo.png';
import "./stylesheets/css/index.css";
import { Grid, Row, Col } from 'react-flexbox-grid';

class App extends Component {

  constructor(props){
    super(props)

    this.state = {
      width: 1080,
      height: 720,
      loading: '',
      results: [],
      ip:'',
      snapshot:null,
      camera_type:'IRYX',
      host:"192.168.1.3",
    }
    window.addEventListener("resize", this.update);
    this.play = this.play.bind(this);
    this.handle_play = this.handle_play.bind(this);
  }

  update = () => {
    this.setState({ 
      width: window.innerWidth, 
      height: window.innerHeight
    })
  }

  handleChange(event) {
    this.setState({"camera_type": event.target.value});
  }

  play (host, encoding = 'h264', pipeline) {
      try {
        console.log('play');

        const videoEl = document.querySelector('video')
        const canvasEl = document.querySelector('canvas')
        // Grab a reference to the video element
        let Pipeline
        let mediaElement
        console.log('ENCODING', encoding)
        if (encoding === 'h264') {
          Pipeline = pipelines.Html5VideoPipeline
          console.log('PIPELINES.HTML5VIDEOPIPELINE', pipelines.Html5VideoPipeline)
          console.log('PIPELINES.HTML5VIDEOPIPELINE', pipelines.Html5VideoPipeline)
          console.log('PIPELINE', Pipeline)
          mediaElement = videoEl
          // hide the other output
          videoEl.style.display = ''
          canvasEl.style.display = 'none'
        } else {
          Pipeline = pipelines.Html5CanvasPipeline
          mediaElement = canvasEl
          // hide the other output
          videoEl.style.display = 'none'
          canvasEl.style.display = ''
        }
        // Setup a new pipeline
        const pipeline = new Pipeline({
          ws: { uri: "ws://"+host+"/rtsp-over-websocket" },
          rtsp: { uri: "rtsp://"+host+"/stream6" },
          mediaElement,
        })

        pipeline.ready.then(() => {
          console.log('pipeline ready')
          pipeline.rtsp.play()
        })
        .catch(e => {
          alert('error ' + e.message )
        })
      } catch (e) {
        console.log('e ', e);
      }
      return pipeline
  }

  handle_play() {
      if (this.state.camera_type === "IRYX"){
          let pipeline;

          pipeline && pipeline.close()

          const device = document.querySelector('#device')
          const host = this.state.ip
          const encoding = "h264"

          console.log(host, encoding)

          // await authorize(host)
          pipeline = this.play(host, encoding, pipeline);
      }
      this.listen_for_events();

  }

  listen_for_events() {
      var ws = new WebSocket("ws://"+this.state.ip+":8091");
      if ("WebSocket" in window) {

          ws.onopen = function() {
              // Web Socket is connected, send data using send()
              console.log("Websocket connected...");
          };
          ws.onmessage = (evt) => { 
              var received_msg = evt.data;
              var results = this.state.results
              results.push(JSON.parse(received_msg));
              if (results.length > 15){
                results.shift();
              }
              console.log(results);
              if (this.state.camera_type === "IRYX"){
                  this.setState({results:results});
              }else{
                  this.setState({results:results, snapshot:"data:image/jpeg;base64,"+JSON.parse(received_msg).snapshot});
              }
          };
          ws.onclose = function() { 
              // websocket is closed.
              alert("Connection is closed..."); 
          };
      } else {
         // The browser doesn't support WebSocket
         alert("WebSocket NOT supported by your Browser!");
      }
  }

  componentDidMount() {
    this.update();
  }
  
  componentWillUnmount() {
    window.removeEventListener('resize', this.update);
  }
 
  render() {
    const videoConstraints = {
      width: this.state.width,
      height: this.state.height,
      facingMode: "user"
    }

    return (

      <Grid fluid>
        <Row>
          <Col xs={12} sm={12} md={12} lg={12}>
            <div className="center-text">
              <img className='logo' src={logo}></img>
            </div>
          </Col>
        </Row>

        <Row center="xs" center="lg" center="md">
          <Col xs={12} sm={12} md={6} lg={6} middle="xs" middle="lg" middle="md">
            <Row center="xs" center="lg" center="md">
                <div>
                  <h3>Live View</h3>
                  <Row center="xs" center="lg" center="md">
                    <Col xs={10} sm={10} md={6} lg={6}>
                      <div className="input-group mb-3">
                        <select value={this.state.camera_type} onChange={(e) => this.handleChange(e)} className="form-control" id="exampleFormControlSelect1">
                            <option value="IRYX" key="IRYX">IRYX</option>
                            <option value="A400" key="A400">A400</option>
                        </select>
                        <input 
                          type="text" 
                          class="form-control" placeholder="Camera IP"
                          onChange={(event) => this.setState({ip: event.target.value})} 
                          value={this.state.ip} />
                      </div>
                    </Col>
                  </Row>
                  <br />
                  
                  <button type="button" 
                  class="btn btn-success" id="play" onClick={this.handle_play}>Start</button>


                  {this.state.snapshot ? <img src={this.state.snapshot} /> : null}
                  <div className='video-holder'>
                    <video
                      style={{width: "100%", height: "100%", }}
                      autoPlay
                    ></video>
                    <canvas style={{width: "100%", height: "100%", }}></canvas>
                  </div>


                </div>
            </Row>
          </Col>
          <Col xs={12} sm={12} md={6} lg={6} 
          middle="xs" middle="lg" middle="md">
            <div className="s-half center-text">
                <h3>Live Events</h3>
                {this.state.results.map((item, key) =>
                    <p>
                      <small key={item.id}> Face Detected <b>{item.face_detected.toString()}</b></small>
                      <small key={item.id}> Eyeduct Visible <b>{item.eyeduct_visible.toString()}</b></small>
                      <small key={item.id}> Temp Measured <b> {(Math.round(item.temperature_measured* 100) / 100).toFixed(2)}</b></small>
                      <small key={item.id}> Avg Temp<b> {(Math.round(item.average_temperature_measured * 100) / 100).toFixed(2)}</b></small>
                    </p>
                )}
            </div>
          </Col>
        </Row>
      </Grid>
      
    )
  }
}

export default App;

