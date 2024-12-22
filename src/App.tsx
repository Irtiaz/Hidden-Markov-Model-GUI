import React, { useState } from 'react';
import { HiddenMarkovModel } from './hmm';

// Types
type Page = 'inputs' | 'models' | 'simulation';

interface State {
  name: string;
  id: number;
}

interface Evidence {
  name: string;
  id: number;
}

const HMMInterface: React.FC = () => {
  // Page state
  const [currentPage, setCurrentPage] = useState<Page>('inputs');
  
  // Input page state
  const [states, setStates] = useState<State[]>([{ name: '', id: 0 }]);
  const [evidences, setEvidences] = useState<Evidence[]>([{ name: '', id: 0 }]);
  
  // Models page state
  const [transitionModel, setTransitionModel] = useState<number[][]>([]);
  const [sensorModel, setSensorModel] = useState<number[][]>([]);
  const [priorProbabilities, setPriorProbabilities] = useState<number[]>([]);
  const [numTimestamps, setNumTimestamps] = useState<number>(10);
  
  // Simulation page state
  const [evidenceSequence, setEvidenceSequence] = useState<number[]>([]);
  const [probabilities, setProbabilities] = useState<number[][]>([]);
  const [likelihood, setLikelihood] = useState<number>(0);
  const [hmm, setHMM] = useState<HiddenMarkovModel | null>(null);

  // Input page handlers
  const addState = () => {
    setStates([...states, { name: '', id: states.length }]);
  };

  const addEvidenceInput = () => {
    setEvidences([...evidences, { name: '', id: evidences.length }]);
  };

  const updateState = (id: number, name: string) => {
    setStates(states.map(state => 
      state.id === id ? { ...state, name } : state
    ));
  };

  const updateEvidence = (id: number, name: string) => {
    setEvidences(evidences.map(evidence => 
      evidence.id === id ? { ...evidence, name } : evidence
    ));
  };

  // Models page handlers
  const updateTransitionModel = (row: number, col: number, value: number) => {
    const newModel = [...transitionModel];
    newModel[row] = [...(newModel[row] || [])];
    newModel[row][col] = value;
    setTransitionModel(newModel);
  };

  const updateSensorModel = (row: number, col: number, value: number) => {
    const newModel = [...sensorModel];
    newModel[row] = [...(newModel[row] || [])];
    newModel[row][col] = value;
    setSensorModel(newModel);
  };

  const updatePriorProbability = (index: number, value: number) => {
    const newPriors = [...priorProbabilities];
    newPriors[index] = value;
    setPriorProbabilities(newPriors);
  };

  // Simulation page handlers
  const addEvidenceToSequence = (evidenceIndex: number) => {
    const newSequence = [...evidenceSequence, evidenceIndex];
    setEvidenceSequence(newSequence);
    
    if (hmm) {
      const newProbabilities = hmm.overall(newSequence, 0, numTimestamps - 1);
      setProbabilities(newProbabilities);
      
      const likelihoods = hmm.likelihoodOfEvidenceSequence(newSequence);
      setLikelihood(likelihoods[likelihoods.length - 1]);
    }
  };

  const renderInputPage = () => (
    <div className="p-4">
      <div className="mb-8">
        <h2 className="text-xl mb-4">States</h2>
        {states.map((state) => (
          <div key={state.id} className="flex mb-2">
            <input
              type="text"
              value={state.name}
              onChange={(e) => updateState(state.id, e.target.value)}
              className="border p-2 mr-2"
              placeholder="Enter state name"
            />
            {state.id === states.length - 1 && (
              <button
                onClick={addState}
                className="bg-blue-500 text-white px-4 py-2 rounded"
              >
                +
              </button>
            )}
          </div>
        ))}
      </div>

      <div className="mb-8">
        <h2 className="text-xl mb-4">Evidences</h2>
        {evidences.map((evidence) => (
          <div key={evidence.id} className="flex mb-2">
            <input
              type="text"
              value={evidence.name}
              onChange={(e) => updateEvidence(evidence.id, e.target.value)}
              className="border p-2 mr-2"
              placeholder="Enter evidence name"
            />
            {evidence.id === evidences.length - 1 && (
              <button
                onClick={addEvidenceInput}
                className="bg-blue-500 text-white px-4 py-2 rounded"
              >
                +
              </button>
            )}
          </div>
        ))}
      </div>

      <button
        onClick={() => setCurrentPage('models')}
        className="bg-green-500 text-white px-6 py-3 rounded"
      >
        Next
      </button>
    </div>
  );

  const renderModelsPage = () => (
    <div className="p-4">
      <div className="mb-8">
        <h2 className="text-xl mb-4">Transition Model</h2>
        <table className="border-collapse">
          <thead>
            <tr>
              <th className="border p-2"></th>
              {states.map((state) => (
                <th key={state.id} className="border p-2">
                  {state.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {states.map((fromState, row) => (
              <tr key={fromState.id}>
                <td className="border p-2">{fromState.name}</td>
                {states.slice(0, -1).map((toState, col) => (
                  <td key={toState.id} className="border p-2">
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      max="1"
                      value={transitionModel[row]?.[col] || ''}
                      onChange={(e) => updateTransitionModel(row, col, parseFloat(e.target.value))}
                      className="w-20 p-1"
                    />
                  </td>
                ))}
                <td className="border p-2">
                  <input
                    type="number"
                    value={1 - (transitionModel[row]?.reduce((a, b) => a + b, 0) || 0)}
                    disabled
                    className="w-20 p-1 bg-gray-100"
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mb-8">
        <h2 className="text-xl mb-4">Sensor Model</h2>
        <table className="border-collapse">
          <thead>
            <tr>
              <th className="border p-2"></th>
              {evidences.map((evidence) => (
                <th key={evidence.id} className="border p-2">
                  {evidence.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {states.map((state, row) => (
              <tr key={state.id}>
                <td className="border p-2">{state.name}</td>
                {evidences.slice(0, -1).map((evidence, col) => (
                  <td key={evidence.id} className="border p-2">
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      max="1"
                      value={sensorModel[row]?.[col] || ''}
                      onChange={(e) => updateSensorModel(row, col, parseFloat(e.target.value))}
                      className="w-20 p-1"
                    />
                  </td>
                ))}
                <td className="border p-2">
                  <input
                    type="number"
                    value={1 - (sensorModel[row]?.reduce((a, b) => a + b, 0) || 0)}
                    disabled
                    className="w-20 p-1 bg-gray-100"
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mb-8">
        <h2 className="text-xl mb-4">Prior Probabilities</h2>
        <table className="border-collapse">
          <tbody>
            {states.slice(0, -1).map((state, index) => (
              <tr key={state.id}>
                <td className="border p-2">{state.name}</td>
                <td className="border p-2">
                  <input
                    type="number"
                    step="0.01"
                    min="0"
                    max="1"
                    value={priorProbabilities[index] || ''}
                    onChange={(e) => updatePriorProbability(index, parseFloat(e.target.value))}
                    className="w-20 p-1"
                  />
                </td>
              </tr>
            ))}
            <tr>
              <td className="border p-2">{states[states.length - 1].name}</td>
              <td className="border p-2">
                <input
                  type="number"
                  value={1 - (priorProbabilities.reduce((a, b) => a + b, 0) || 0)}
                  disabled
                  className="w-20 p-1 bg-gray-100"
                />
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="mb-8">
        <h2 className="text-xl mb-4">Number of Timestamps</h2>
        <input
          type="number"
          min="1"
          value={numTimestamps}
          onChange={(e) => setNumTimestamps(parseInt(e.target.value))}
          className="border p-2"
        />
      </div>

      <button
        onClick={() => {
          const model = new HiddenMarkovModel(
            transitionModel,
            sensorModel,
            priorProbabilities
          );
          setHMM(model);
          setCurrentPage('simulation');
        }}
        className="bg-green-500 text-white px-6 py-3 rounded"
      >
        Next
      </button>
    </div>
  );

  const renderSimulationPage = () => (
    <div className="flex p-4">
      <div className="w-1/2 pr-4">
        <h2 className="text-xl mb-4">Probabilities</h2>
        <table className="border-collapse">
          <thead>
            <tr>
              <th className="border p-2">Timestamp</th>
              {states.map((state) => (
                <th key={state.id} className="border p-2">
                  {state.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {probabilities.map((row, timestamp) => (
              <tr key={timestamp}>
                <td className="border p-2">{timestamp + 1}</td>
                {row.map((prob, index) => (
                  <td key={index} className="border p-2">
                    {prob.toFixed(4)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="w-1/2 pl-4">
        <h2 className="text-xl mb-4">Evidence Controls</h2>
        <div className="mb-4">
          {evidences.map((evidence) => (
            <button
              key={evidence.id}
              onClick={() => addEvidenceToSequence(evidence.id)}
              className="bg-blue-500 text-white px-4 py-2 rounded mr-2 mb-2"
            >
              {evidence.name}
            </button>
          ))}
        </div>
        
        <div className="mb-4">
          <h3 className="text-lg mb-2">Evidence Sequence</h3>
          <div className="p-2 border rounded">
            {evidenceSequence.map((evidenceIndex) => 
              evidences[evidenceIndex].name
            ).join(' → ')}
          </div>
        </div>

        <div>
          <h3 className="text-lg mb-2">Likelihood</h3>
          <div className="p-2 border rounded">
            {likelihood.toFixed(6)}
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="container mx-auto">
      {currentPage === 'inputs' && renderInputPage()}
      {currentPage === 'models' && renderModelsPage()}
      {currentPage === 'simulation' && renderSimulationPage()}
    </div>
  );
};

export default HMMInterface;
