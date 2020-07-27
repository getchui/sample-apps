import React from 'react';

const Result = (props) => {

  return (
    <div>
        <h3>Result: {props.result}</h3>
        <h4>Real {props.real.toFixed(2)}</h4>
        <h4>Fake {props.fake.toFixed(2)}</h4>
        <input className="annotator" value={props.annotator} onChange={props.onChange} />
        <h4>Is this result correct?</h4>
        <span onClick={() => props.postfunction(true)}>Yes </span><span onClick={() => props.postfunction(false)}>No</span>
    </div>
  )
  
}

export default Result;