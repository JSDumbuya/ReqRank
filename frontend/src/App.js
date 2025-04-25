import './App.css';
import React, { use, useEffect, useState } from "react";
import M from "materialize-css";



function App() {
  
  const [requirementFile, setRequirementFile] = useState(null);
  const [requirements, setRequirements] = useState([]);
  const [includeCost, setIncludeCost] = useState(false);
  const [includeEffort, setIncludeEffort] = useState(false);
  const [prioritizedData, setPrioritizedData] = useState([]);
  const [stakeholders, setStakeholders] = useState([]);
  const [isStakeholdersPrioritized, setisStakeholdersPrioritized] = useState(false);
  const [errors, setErrors] = useState([]);

  useEffect(() => {
    M.Collapsible.init(document.querySelectorAll(".collapsible"));
  }, [stakeholders]);

  const handleRquirementFileUpload = (event) => {
    const file = event.target.files[0];
    setRequirementFile(file);
  
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      const lines = text.split(/\r?\n/).slice(1); 
  
      const parsed = lines
        .map(line => line.trim())
        .filter(line => line.length > 0)
        .map(line => {
          const [text, cost, effort] = line.split(",");
          return {
            text: text?.replaceAll('"', '').trim(),
            cost: cost ? parseFloat(cost.trim()) : null,
            effort: effort ? parseFloat(effort.trim()) : null,
          };
        });
  
      setRequirements(parsed);
    };
  
    reader.readAsText(file);
  };
  

  const handleAddStakeholder = () => {
    setStakeholders([...stakeholders, { name: "", file: null }])
  };

  const handleUpdateStakeholderName = (index, newName) => {
    const updateStakeholders = [...stakeholders];
    updateStakeholders[index].name = newName;
    setStakeholders(updateStakeholders);
  };

  const handleStakeholderFileUpload = (index, event) => {
    const updateStakeholders = [...stakeholders];
    updateStakeholders[index].file = event.target.files[0];
    setStakeholders(updateStakeholders);
  };

  const handleChangeStakeholderPriority = (index, direction) => {
    const updateStakeholders = [...stakeholders];
    const newIndex = index + direction;

    if (newIndex >= 0 && newIndex < stakeholders.length) {
      [updateStakeholders[index], updateStakeholders[newIndex]] = 
      [updateStakeholders[newIndex], updateStakeholders[index]];
      setStakeholders(updateStakeholders);
    } 
  };

  const handleRemoveStakeholder = (index) => {
    setStakeholders(stakeholders.filter((_, i) => i !== index));
  };

  const handlePrioritizeRequirements = async () => {
    let newErrors = [];
    if (!requirementFile) {
      newErrors.push("Please upload a file with the requirements you want prioritized.");
    }
    if (stakeholders.length === 0) {
      newErrors.push("Please add a least one stakeholder.");
    } else {
      stakeholders.forEach((stakeholder, index) => {
        if (!stakeholder.name || !stakeholder.file) {
          newErrors.push(`Stakeholder ${index + 1} is missing ${!stakeholder.name ? "a name" : ""}${!stakeholder.name && !stakeholder.file ? " and " : ""}${!stakeholder.file ? "a file" : ""}.`);
        }
      });
    }
    if (newErrors.length > 0) {
      setErrors(newErrors);
      return;
    }

    setErrors([]);

    //insert api call here to set prioritized data - use flask
    //Send requirements (the list) with the post call
    {/*
      fetch("/api/prioritize", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ requirements })
});

      */}
  };

  return (
    <div className="App">

      <h1 className="blue-text text-darken-2 center-align">ReqRank</h1>

      {/* Displaying prioritization results */}
      <div className="card-panel grey lighten-4">
        <h5 className="center-align">Prioritized Requirements</h5>
        <div style={{ maxHeight: "400px", overflow: "auto"}}>
          <table className="highlight centered">
            <thead>
              <tr>
                <th>Requirement</th>
                <th>Priority Score</th>
                <th>Group</th>
              </tr>
            </thead>
            <tbody>
              {prioritizedData.length > 0 ? (
                prioritizedData.map((item, index) => (
                  <tr key={index}>
                    <td>{item.requirement}</td>
                    <td>{item.priority_score}</td>
                    <td>{item.group}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="5">No results available yet.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/*Requirements file upload field*/}
      <ul className='collapsible popout'>
        <li>
          <div className='collapsible-header'>
            <i className='material-icons'>description</i>
            Upload Requirements
          </div>
          <div className='collapsible-body'>
            {/*Requirements file upload field*/}
            <div className='file-field input-field'>
              <div className='btn blue'>
                <span>Upload file</span>
                <input
                type='file'
                onChange={(e) => handleRquirementFileUpload(e)}/>
              </div>
              <div className='file-path-wrapper'>
                <input
                className='file-path validate'
                type='text'
                placeholder='Upload document'
                value={requirementFile ? requirementFile.name : ""}
                readOnly />
              </div>
            </div>
            {/*Requirements file upload field*/}

            {/*Display uploaded requirements*/}
            {requirements.length > 0 && (
              <div style={{ maxHeight: "200px", overflowY: "scroll", marginTop: "1rem" }}>
                <table className="striped responsive-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Requirement</th>
                      {includeCost && <th>Cost</th>}
                      {includeEffort && <th>Effort</th>}
                    </tr>
                  </thead>
                  <tbody>
                    {requirements.map((req, index) => (
                      <tr key={index}>
                        <td>{index + 1}</td>
                        <td>{req.text}</td>
                        {includeCost && (
                          <td>
                            <input
                              type="number"
                              value={req.cost ?? ""}
                              onChange={(e) => {
                                const newValue = e.target.value;
                                const updated = [...requirements];
                                updated[index].cost = newValue !== "" ? parseFloat(newValue) : null;
                                setRequirements(updated);
                              }}
                            />
                          </td>
                        )}
                        {includeEffort && (
                          <td>
                            <input
                              type="number"
                              value={req.effort ?? ""}
                              onChange={(e) => {
                                const newValue = e.target.value;
                                const updated = [...requirements];
                                updated[index].effort = newValue !== "" ? parseFloat(newValue) : null;
                                setRequirements(updated);
                              }}
                            />
                          </td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {/*Display uploaded requirements*/}

            {/*Additional criteria*/}
            <div className="section">
            <h5>Add Additional Prioritization Criteria</h5>
            <p><strong>
            To include cost and/or effort for each requirement, check the boxes below. <br/>
            You can upload a CSV file that already contains this information, or add it manually after the upload.
            </strong></p>
            <label>
              <input type="checkbox" checked={includeCost} onChange={() => setIncludeCost(!includeCost)} />
              <span style={{fontSize: '20px'}}><strong>Include Cost</strong></span>
            </label>
            <br/>
            <label>
              <input type="checkbox" checked={includeEffort} onChange={() => setIncludeEffort(!includeEffort)} />
              <span style={{fontSize: '20px'}}><strong>Include Effort</strong></span>
            </label>
          </div>
          {/*Additional criteria*/}
          </div>
        </li>
      </ul>

      {/*Requirements file upload field*/}

      {/*Row with add stakeholder button + are stakeholders prioritized checkbox */}
      <div className='row'>
        <div className='col s12 m6'>
          {/*Add stakeholder button*/}
          <button className='waves-effect waves-light btn-large' onClick={handleAddStakeholder}>
            <i className='material-icons right'>add</i>
            Add Stakeholder
          </button>
        </div>

        <div className='col s12 m6'>
          {/*Checkbox: prioritized?*/}
          <div className="input-field">
            <label>
            <input type="checkbox" checked={isStakeholdersPrioritized} onChange={() => setisStakeholdersPrioritized(!isStakeholdersPrioritized)}/>
            <span>Stakeholders are listed in order of importance</span>
            </label>
          </div>
        </div>
      </div>

      {/*Row with add stakeholder button + are stakeholders prioritized checkbox */}

      {/*Stakeholder list and files*/}
      <ul className='collapsible popout'>
        {stakeholders.map((stakeholder, index) => (
          <li key={index}>
            <div className='collapsible-header'>
              <i className='material-icons'>person</i>
              {stakeholder.name || `Stakeholder ${index + 1}`}
              <div className='right' style={{ marginLeft: '10px'}}>
                <button 
                className='btn-small yellow darken-3' 
                disabled={index === 0} 
                style={{ marginLeft: '5px'}}
                onClick={(e) => {e.stopPropagation(); handleChangeStakeholderPriority(index, -1); }}>
                  <i className='material-icons'>arrow_upward</i>
                </button>
                <button
                className='btn-small yellow darken-3'
                disabled={index === stakeholder.length - 1}
                style={{ marginLeft: '5px'}}
                onClick={(e) => {e.stopPropagation(); handleChangeStakeholderPriority(index, 1); }}>
                  <i className='material-icons'>arrow_downward</i>
                </button>
                <button
                className='btn-small red'
                style={{ marginLeft: '5px'}}
                onClick={(e) => {e.stopPropagation(); handleRemoveStakeholder(index); }}>
                  <i className='material-icons'>delete</i>
                </button>
              </div>
            </div>

            <div className='collapsible-body'>
              <div className='input-field'>
                <input 
                type='text' 
                value={stakeholder.name} 
                onChange={(e) => handleUpdateStakeholderName(index, e.target.value)}
                placeholder='Enter stakeholder name'/>
                <label className='active'>Stakeholder Name</label>
              </div>

              <div className='file-field input-field'>
                <div className='btn blue'>
                  <span>Upload file</span>
                  <input
                  type='file'
                  onChange={(e) => handleStakeholderFileUpload(index, e)}/>
                </div>
                <div className='file-path-wrapper'>
                  <input
                  className='file-path validate'
                  type='text'
                  placeholder='Upload document'
                  value={stakeholder.file ? stakeholder.file.name : ""}
                  readOnly/>
                </div>
              </div>
            </div>
          </li>
        ))}
      </ul>
      {/*Stakeholder list and files*/}
        
      {/* Display error messages + prioritization button*/}
      <div>
        {errors.length > 0 && (
          <div className='card-panel red lighten-4 red-text text-darken-4'>
            <i className='material-icons right'>error</i>
            <ul>
              {errors.map((error, index) => (
                <div key={index}>
                   <li>{error}</li>
                   {index < errors.length - 1 && <hr />}
                </div>
              ))}
            </ul>
          </div>)}
        
        {/* Display error messages */}

        {/* Prioritization button */}
        <button type="button" className='waves-effect waves-light btn-large' onClick={handlePrioritizeRequirements}>
          <i className="material-icons right">send</i>
          Prioritize Requirements
        </button>
        {/* Prioritization button */}
      </div>

      {/* Display error messages + prioritization button*/}
      

    </div>
  );
}

export default App;


