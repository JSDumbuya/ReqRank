import './App.css';
import React, { useEffect, useState, useRef } from "react";
import M from "materialize-css";


function App() {
  
  const [requirementFile, setRequirementFile] = useState(null); //to display file in interface
  const [requirements, setRequirements] = useState([]); //actual req list
  const [includeCost, setIncludeCost] = useState(false);
  const [includeEffort, setIncludeEffort] = useState(false);
  const [prioritizedData, setPrioritizedData] = useState([]);
  const [stakeholders, setStakeholders] = useState([]);
  const [isStakeholdersPrioritized, setisStakeholdersPrioritized] = useState(false);
  const [selectedExplanation, setSelectedExplanation] = useState('');
  const [errors, setErrors] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [weights, setWeights] = useState({
    cost: 5,
    effort: 5,
    amountRelatedReqs: 5,
    sentiment: 5,
    popularity: 5,
    nfrImportance: 5
  })
  const [nfrWeights, setNfrWeights] = useState({
    PE: 5, 
    US: 5,
    SE: 5, 
    A: 5, 
    MN: 5, 
    L: 5,  
    SC: 5, 
    LF: 5, 
    O: 5,  
    FT: 5,
    PO: 5,
  });
  const nfrLabels = {
    PE: "Performance",
    US: "Usability",
    SE: "Security",
    A: "Availability",
    MN: "Maintainability",
    L: "Legal",
    SC: "Scalability",
    LF: "Look-and-Feel",
    O: "Operability",
    FT: "Fault Tolerance",
    PO: "Portability"
  };
  const nfrDefinitions = {
    SE: "Protecting data and systems from unauthorized access or harm.",
    US: "Making the system easy and pleasant for people to use and understand.",
    O:  "Ensuring the system runs smoothly and can be managed easily.",
    PE: "How fast and responsive the system is during use.",
    LF: "The visual style and overall experience of using the system.",
    A:  "Making sure the system is up and running when people need it.",
    MN: "How easy it is to update, fix, or improve the system.",
    SC: "The ability to handle more users or work without problems.",
    FT: "The system’s ability to keep working even when something goes wrong.",
    L:  "Following laws and rules that apply to the system and its use.",
    PO: "How easily the system can work on different devices or environments."
  };
  
  const tooltipInitialized = useRef(false);

  useEffect(() => {
    M.Collapsible.init(document.querySelectorAll(".collapsible"));
    M.Modal.init(document.querySelectorAll('.modal'));

    if (!tooltipInitialized.current) {
      M.Tooltip.init(document.querySelectorAll(".tooltipped"));
      tooltipInitialized.current = true;
    }
  
  }, [stakeholders, prioritizedData]);


  //Requirement file upload

  const handleRquirementFileUpload = (event) => {
    const file = event.target.files[0];
  
    if (!file || !(file instanceof Blob)) {
      alert("Please select a valid file.");
      return;
    }
  
    setRequirementFile(file);
  
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      const lines = text.split(/\r?\n/);
  
      const parsed = lines
        .map(line => line.trim())
        .filter(line => line.length > 0)
        .map(line => ({
          text: line.replaceAll('"', ''),
        }));
  
      setRequirements(parsed);
    };
  
    reader.readAsText(file);
  };
  
  // Configuration of prioritization

  const handleCriteriaWeightChange = (name, value) => {
    setWeights((prev) => ({
      ...prev,
      [name]: Number(value),
    }));
  };

  const handleNfrWeightChange = (key, value) => {
    setNfrWeights(prev => ({
      ...prev,
      [key]: Number(value)
    }));
  };  
  
  const normalizeWeights = (weights) => {
    const total = Object.values(weights).reduce((sum, val) => sum + val, 0);
    const normalized = {};
    for (const key in weights) {
      normalized[key] = weights[key] / total;
    }
    return normalized;
  };

  //Stakeholder related

  const handleAddStakeholder = () => {
    setStakeholders([...stakeholders, { name: "", file: null, feedback: "", weight: stakeholders.length + 1 }])
  };

  const handleUpdateStakeholderName = (index, newName) => {
    const updateStakeholders = [...stakeholders];
    updateStakeholders[index].name = newName;
    setStakeholders(updateStakeholders);
  };

 /*  const handleStakeholderFileUpload = (index, event) => {
    const updateStakeholders = [...stakeholders];
    updateStakeholders[index].file = event.target.files[0];
    setStakeholders(updateStakeholders);
  }; */

  const handleStakeholderFeedbackChange = (index, value) => {
    const updatedStakeholders = [...stakeholders];

    const textFile = new File([value], `stakeholder_${index}_feedback.txt`, {
      type: 'text/plain',
    });
  
    updatedStakeholders[index].feedback = value;
    updatedStakeholders[index].file = textFile;
  
    setStakeholders(updatedStakeholders);
  };

  const handleChangeStakeholderPriority = (index, direction) => {
    const updatedStakeholders = [...stakeholders];
    const newIndex = index + direction;
  
    if (newIndex >= 0 && newIndex < stakeholders.length) {
      [updatedStakeholders[index], updatedStakeholders[newIndex]] =
        [updatedStakeholders[newIndex], updatedStakeholders[index]];
  
      setStakeholders(updateStakeholderWeights(updatedStakeholders));
    }
  };

  const updateStakeholderWeights = (stakeholdersList) => {
    const total = stakeholdersList.length;
    return stakeholdersList.map((stakeholder, index) => ({
      ...stakeholder,
      weight: total - index,
    }));
  };

  const handleRemoveStakeholder = (index) => {
    setStakeholders(stakeholders.filter((_, i) => i !== index));
  };

  //Prioritize reqs and display reqs

  const handlePrioritizeRequirements = async () => {
    let newErrors = [];
  
    if (requirements.length === 0) {
      newErrors.push("Please upload a requirements file.");
    }
    if (stakeholders.length === 0) {
      newErrors.push("Please add at least one stakeholder.");
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
    setIsLoading(true);
  
    const formData = new FormData();
    
    formData.append("requirements", JSON.stringify(requirements));
    formData.append("normalizedWeights", JSON.stringify(normalizeWeights(weights)));
    formData.append("normalizedNfrWeights", JSON.stringify(normalizeWeights(nfrWeights)));
    formData.append("stakeholdersPrioritized", JSON.stringify(isStakeholdersPrioritized));
  
    stakeholders.forEach((stakeholder, index) => {
      formData.append(`stakeholderFile`, stakeholder.file);
      formData.append(`stakeholderNames`, stakeholder.name); 
      formData.append(`stakeholderWeights`, stakeholder.weight); 
    });
  
    try {
      const response = await fetch("http://localhost:8000/api/prioritize", {
        method: "POST",
        body: formData,
      });
  
      if (response.ok) {
        const data = await response.json();
        setPrioritizedData(data.prioritized_data);
      } else {
        const errorData = await response.json();
        alert(`Error: ${errorData.message || "Something went wrong!"}`);
      }
    } catch (error) {
      alert(`Network Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleShowExplanation = (explanation) => {
    setSelectedExplanation(explanation);
    const modalElem = document.getElementById('explanation-modal');
    const modalInstance = M.Modal.getInstance(modalElem);
    modalInstance.open();
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
                    <td onClick={() => handleShowExplanation(item.explanation)} style={{ cursor: 'pointer' }}>{item.text}</td>
                    <td>{item.final_score.toFixed(2)}</td>
                    <td>{item.group_nr}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="5">No results available yet.</td>
                </tr>
              )}
            </tbody>
          </table>

          <div id="explanation-modal" className="modal">
            <div className="modal-content">
              <h4>Scoring Explanation</h4>
              <p style={{ whiteSpace: 'pre-line' }}>{selectedExplanation}</p>
            </div>
            <div className="modal-footer">
              <a href="#!" className="modal-close waves-effect waves-green btn-flat">Close</a>
            </div>
          </div>

        </div>
      </div>
      {/* Displaying prioritization results */}

      <h4>Upload Requirements & Set Prioritization Rules</h4>

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
            To include cost and/or effort for each requirement, check the boxes below. 
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
      
      {/*Configuration of requirements prioritization*/}
      <ul className='collapsible popout'>
        <li>
          <div className='collapsible-header'>
            <i className='material-icons'>tune</i>
            Set Prioritization Rules
          </div>
          <div className='collapsible-body'>
            {/* Weighting criteria */}
            <h5>Apply Weights to Prioritization Criteria</h5>
            <p>Set the weights of the following criteria according to their importance for the prioritization of requirements.</p>
            <form className='container'>
              {includeCost && (
                <div className="input-field">
                  <p className="range-field">
                    <label htmlFor="costWeight" className="active">
                      Cost Importance: {weights.cost}
                    </label>
                    <input
                      type="range"
                      id="costWeight"
                      min="0"
                      max="10"
                      step="1"
                      value={weights.cost}
                      onChange={(e) => handleCriteriaWeightChange("cost", e.target.value)}
                    />
                  </p>
                </div>
              )}

              {includeEffort && (
                <div className="input-field">
                  <p className="range-field">
                    <label htmlFor="effortWeight" className="active">
                      Effort Importance: {weights.effort}
                    </label>
                    <input
                      type="range"
                      id="effortWeight"
                      min="0"
                      max="10"
                      step="1"
                      value={weights.effort}
                      onChange={(e) => handleCriteriaWeightChange("effort", e.target.value)}
                    />
                  </p>
                </div>
              )}

              <div className="input-field">
                <p className="range-field">
                  <label htmlFor="amountRelatedReqsWeight" className="active">
                    Amount of Related Requirements Importance: {weights.amountRelatedReqs}
                  </label>
                  <input
                    type="range"
                    id="amountRelatedReqsWeight"
                    min="0"
                    max="10"
                    step="1"
                    value={weights.amountRelatedReqs}
                    onChange={(e) => handleCriteriaWeightChange("amountRelatedReqs", e.target.value)}
                  />
                </p>
              </div>

              <div className="input-field">
                <p className="range-field">
                  <label htmlFor="sentimentWeight" className="active">
                    Sentiment Importance: {weights.sentiment}
                  </label>
                  <input
                    type="range"
                    id="sentimentWeight"
                    min="0"
                    max="10"
                    step="1"
                    value={weights.sentiment}
                    onChange={(e) => handleCriteriaWeightChange("sentiment", e.target.value)}
                  />
                </p>
              </div>

              <div className="input-field">
                <p className="range-field">
                  <label htmlFor="popularityWeight" className="active">
                    Popularity Among Stakeholders Importance: {weights.popularity}
                  </label>
                  <input
                    type="range"
                    id="popularityWeight"
                    min="0"
                    max="10"
                    step="1"
                    value={weights.popularity}
                    onChange={(e) => handleCriteriaWeightChange("popularity", e.target.value)}
                  />
                </p>
              </div>

              <div className="input-field">
                <p className="range-field">
                  <label htmlFor="nfrImportance" className="active">
                    Influence of NFR Category Weights: {weights.nfrImportance}
                  </label>
                  <input
                    type="range"
                    id="nfrImportance"
                    min="0"
                    max="10"
                    step="1"
                    value={weights.nfrImportance}
                    onChange={(e) => handleCriteriaWeightChange("nfrImportance", e.target.value)}
                  />
                </p>
              </div>
            </form>
            {/* Weighting criteria */}

            <div className="divider"></div>

          {/* NFR ranking */}
          <div className="section">
            <h5>Rate Non-Functional Requirements by Importance</h5>
            <p>All NFRs are treated equally unless their importance is adjusted. <br></br> Tip: Hover over each NFR name to see a short definition.</p>
            <form className="container">
              {Object.keys(nfrWeights).map((key) => (
                <div key={key} className="input-field">
                  <p className="range-field">
                    <label
                      htmlFor={key}
                      className="active tooltipped"
                      data-tooltip={nfrDefinitions[key]}>
                      {nfrLabels[key]} Importance: {nfrWeights[key]}
                    </label>
                    <input
                      type="range"
                      id={key}
                      min="0"
                      max="10"
                      step="1"
                      value={nfrWeights[key]}
                      onChange={(e) => handleNfrWeightChange(key, e.target.value)}
                    />
                  </p>
                </div>
              ))}
            </form>
          </div>
          {/* NFR ranking */}
          </div>
        </li>
      </ul>
      {/*Configuration of requirements prioritization*/}

      <h4>Stakeholder Feedback & Requirements Prioritization</h4>

      <p>
        Add stakeholders and upload their feedback. If you'd like to prioritize certain stakeholders, check the box below. <br/>
        You can adjust their priority using the arrow buttons.
      </p>


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
                className='btn-small yellow darken-1' 
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
              
              <div className='input-field'>
                <label className='active'>Stakeholder Feedback</label>
                <textarea
                  className='materialize-textarea'
                  value={stakeholder.feedback || ''}
                  onChange={(e) => handleStakeholderFeedbackChange(index, e.target.value)}
                  placeholder='Paste or type stakeholder feedback here...'
                  style={{ minHeight: '150px', maxHeight: '300px', overflowY: 'scroll' }}/>
              </div>

              {/*<div className='file-field input-field'>
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
              </div>*/}
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
        {isLoading && (
          <div className="card-panel yellow lighten-4 amber-text text-darken-4">
          <i className="material-icons right">hourglass_empty</i>
          Prioritization is running. Please wait...
        </div>)}
        <button type="button" className='waves-effect waves-light btn-large' onClick={handlePrioritizeRequirements} disabled={isLoading}>
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


