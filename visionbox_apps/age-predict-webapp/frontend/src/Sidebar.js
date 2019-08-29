import React from "react";
import logo from './assets/logo.png';

const Sidebar = () => {
  return (
    <div className="sidebar">
      <img src={logo} alt='logo'/>
      <h3>AgeBox</h3>
      <a href="https://support.trueface.ai">Support</a>
    </div>
  )
}

export default Sidebar;