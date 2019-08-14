import React from 'react';

const Result = (props) => {
  
  if (props.age === false){
    return (
      <div>
        <h3>No face found</h3>
      </div>
    )
  }

  return (
    <div>
        <h3>Age Prediction:</h3>
        <h2>{props.age.toFixed(0)}</h2>
    </div>
  )
  
}

export default Result;