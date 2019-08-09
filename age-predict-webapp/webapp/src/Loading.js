import React from 'react';
import loading from "./loading.svg";
 
const Loading = () => (
   <div className="spinner">
       <img src={loading} alt="Loading" />
   </div>
);
 
export default Loading;