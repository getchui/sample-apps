import React, { Component } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import Loading from './Loading';
import Result from './Result';
 
class App extends Component {
  constructor(props){
    super(props)
    this.state = {
      width: 720,
      height: 480,
      age: '',
      loading: '',
      result: '',
      test: '',
    }
    this.setRef = this.setRef.bind(this)
    this.capture = this.capture.bind(this)
    window.addEventListener("resize", this.update)
  }

  setRef = webcam => {
    this.webcam = webcam;
  }

  submitPhoto = photo => {
    this.setState({
      loading: true
    })

    const axiosConfig = {
      headers: {
          'Content-Type': 'application/json',
          "Access-Control-Allow-Origin": "*",
      }
    }

    const photoData = {
      image: photo
    }

    axios.post('http://localhost:9001', photoData, axiosConfig)
      .then(response => {
        if (response.success) {
          this.setState({
            age: response.data[0].estimated_age,
            result: response.success,
            loading: false
          })
        } else {
          this.setState({
            result: response.success,
            loading: false
          })
        }
      })
      .catch((err) => {
        console.log(err)
      })
  }
 
  capture = () => {
    const imageSrc = this.webcam.getScreenshot();
    this.submitPhoto(imageSrc)
  }

  update = () => {
    this.setState({ 
      width: window.innerWidth, 
      height: window.innerHeight
    })
  }

  componentDidMount() {
    this.update()
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
      <div className="webcam">
        <Webcam
          audio={false}
          height={this.state.height}
          ref={this.setRef}
          screenshotFormat="image/jpeg"
          width={this.state.width}
          videoConstraints={videoConstraints}
        />
        <button onClick={this.capture}>Capture photo</button>
        { this.state.loading && <Loading className="loading" type="spinningBubbles" color="#fff" /> }
        { this.state.result && <Result /> }
      </div>
    );
  }
}

export default App;