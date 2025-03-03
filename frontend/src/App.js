import './App.css';
import React, { useState } from "react";



function App() {
  const [feedbackFiles, setFeedbackFiles] = useState([]);
  const [requirementFile, setRequirementFile] = useState(null);

  const handleUserFeedbackData = (event) => {
    const newFiles = Array.from(event.target.files);
    setFeedbackFiles((prevFiles) => [...prevFiles, ...newFiles]);
  };

  const handleRquirementFileData = (event) => {
    const file = event.target.files[0];
    setRequirementFile(file);
  };


  const removeFeedbackFile = (index) => {
    setFeedbackFiles(feedbackFiles.filter((_, i) => i !== index));
  };

  return (
    <div className="App">
      <h1 className="blue-text text-darken-2 center-align">ReqRank</h1>
      {/* Div for displaying prioritization results */}
      <div className="card-panel grey lighten-4">
        <h5 className="center-align">Prioritized Results</h5>
        <p className="center-align">Results will be displayed here.</p>
      </div>

      {/* File uploads */}
      <div className="file_upload_container">
        <div className="row">
          {/* User feedback field */}
          <div className="col s12 m6 l6">
            <div className="feedback_input_field">
              <label className="active">Upload user feedback</label>
              <input type="file" multiple onChange={handleUserFeedbackData} />
            </div>

            <ul className="collection">
              {feedbackFiles.map((file, index) => (
                <li key={index} className="collection-item">
                  {file.name}
                  <button className="btn red btn-small right" onClick={() => removeFeedbackFile(index)}>Remove</button>
                </li>
              ))}
            </ul>
          </div>

          {/* Requirements input field */}
          <div className="col s12 m6 l6">
            <div className="req_input_field">
              <label className="active">Upload Requirements</label>
              <input type="file" onChange={handleRquirementFileData} />
            </div>

            {requirementFile && (
              <div className="card-panel">
                <p>Uploaded file: {requirementFile.name}</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Prioritization button */}
      <div>
        <button type="button" className='waves-effect waves-light btn-large'><i className="material-icons right">send</i>
        Prioritize Requirements</button>
      </div>
    </div>
  );
}

export default App;


