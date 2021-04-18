import React from 'react';
import './helper.css'
const TriggerButton = ({ triggerText, buttonRef, showModal }) => {
  return (
    <button
      className="createJobButton"
      ref={buttonRef}
      onClick={showModal}
    >
      {triggerText}
    </button>
  );
};
export default TriggerButton;