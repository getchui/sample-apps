import React, { Component } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import Result from './Result';
import Loading from './Loading';
import SideBar from './Sidebar';

import "./stylesheets/css/index.css";
 
class App extends Component {
  constructor(props){
    super(props)
    this.state = {
      width: 1080,
      height: 720,
      age: false,
      loading: '',
      result: false,
      test: '',
    }
    this.setRef = this.setRef.bind(this)
    this.capture = this.capture.bind(this)
    window.addEventListener("resize", this.update)
    this.submitPhoto = this.submitPhoto.bind(this)
  }

  setRef = webcam => {
    this.webcam = webcam;
  }

  submitPhoto = photo => {
    this.setState({
      loading: true
    })

    const photoData = {
      image: photo
    }

    axios.post('/predict', photoData)
      .then(response => {
        console.log(response)
        if(response.data.success){
          this.setState({
            result: true,
            age: response.data.data[0].estimated_age,
            loading: false,
          })
        } else {
          this.setState({
            result: true,
            age: false,
            loading: false,
          })
        }
      })
      .catch((err) => {
        console.log(err)
      })
  }
 
  capture = () => {
    const imageSrc = this.webcam.getScreenshot();
    
    this.submitPhoto(imageSrc.slice(23))

    setTimeout(() => {
      document.querySelector('.results').classList.add('fade')
    }, 5000)
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
      <div className='main'>
        <SideBar />
        <div className="webcam">
          <Webcam
            audio={false}
            height={this.state.height}
            ref={this.setRef}
            screenshotFormat="image/jpeg"
            width={this.state.width}
            videoConstraints={videoConstraints}
          />
          <button onClick={this.capture}></button>
          <div>{ this.state.loading  && <Loading />}</div>
          <div className='results'>{ this.state.result && <Result age={this.state.age} /> }</div>
        </div>
      </div>
    )
  }
}

export default App;