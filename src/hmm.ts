export class HiddenMarkovModel {
	private readonly transitionModel: number[][];
	private readonly sensorModel: number[][];
	private readonly priorProbabilities: number[];

	constructor(transitionModel: number[][], sensorModel: number[][], priorProbability: number[]) {
		if (!this.transitionModelIsValid(transitionModel)) {
			throw new Error("Invalid transition model");
		}

		if (!this.sensorModelIsValid(sensorModel)) {
			throw new Error("Invalid sensor model");
		}

		if (!this.priorProbabilityIsValid(priorProbability, transitionModel.length)) {
			throw new Error("Invalid prior probability");
		}

		this.transitionModel = transitionModel.map(row => row.concat([1 - row.reduce((accumulator, currentValue) => accumulator + currentValue)]));
		this.sensorModel = sensorModel.map(row => row.concat([1 - row.reduce((accumulator, currentValue) => accumulator + currentValue)]));

		this.priorProbabilities = priorProbability.concat([1 - priorProbability.reduce((accumulator, currentValue) => accumulator + currentValue)]);
	}
	
	private likelihoodOfEvidenceSequenceAndLastState(evidenceSequence: number[]): number[][] {
		if (!this.evidenceSequenceIsValid(evidenceSequence)) {
			throw new Error(`Evidence sequence - ${evidenceSequence.toString()} is invalid`);
		}

		const totalStates = this.transitionModel.length;
		const L: number[][] = this.defineBlank2DArray(evidenceSequence.length, totalStates);

		for (let t = 0; t < evidenceSequence.length; ++t) {
			const currentEvidence = evidenceSequence[t];

			for (let state = 0; state < totalStates; ++state) {

				if (t === 0) {
					L[t][state] = this.priorProbabilities[state] * this.sensorModel[state][currentEvidence];
				}

				else {
					L[t][state] = 0;
					for (let previousState = 0; previousState < totalStates; ++previousState) {
						L[t][state] += L[t - 1][previousState] * this.transitionModel[previousState][state] * this.sensorModel[state][currentEvidence];
					}
				}

			}
		}

		return L;
	}

	public likelihoodOfEvidenceSequence(evidenceSequence: number[]): number[] {
		const totalStates = this.transitionModel.length;
		const likelihoodOfEvidenceSequenceAndLastState = this.likelihoodOfEvidenceSequenceAndLastState(evidenceSequence);
		
		const likelihoods: number[] = new Array(evidenceSequence.length);

		for (let t = 0; t < evidenceSequence.length; ++t) {
			likelihoods[t] = 0;
			for (let lastState = 0; lastState < totalStates; ++lastState) {
				likelihoods[t] += likelihoodOfEvidenceSequenceAndLastState[t][lastState];
			}
		}

		return likelihoods;
	}

	public filtering(evidenceSequence: number[]): number[][] {
		if (!this.evidenceSequenceIsValid(evidenceSequence)) {
			throw new Error(`Evidence sequence - ${evidenceSequence.toString()} is invalid`);
		}

		const totalStates = this.transitionModel.length;
		const f: number[][] = this.defineBlank2DArray(evidenceSequence.length, totalStates);
		
		for (let t = 0; t < evidenceSequence.length; ++t) {
			const currentEvidence = evidenceSequence[t];

			for (let state = 0; state < totalStates; ++state) {

				if (t === 0) {
					f[t][state] = this.priorProbabilities[state] * this.sensorModel[state][currentEvidence];
				}

				else {
					f[t][state] = 0;
					for (let previousState = 0; previousState < totalStates; ++previousState) {
						f[t][state] += f[t - 1][previousState] * this.transitionModel[previousState][state] * this.sensorModel[state][currentEvidence];
					}
				}
			}

			f[t] = this.normalize(f[t]);
		}
		
		return f;
	}

	public prediction(evidenceSequence: number[], futureTimeStamp: number): number[][] {
		if (!this.evidenceSequenceIsValid(evidenceSequence)) {
			throw new Error("Invalid evidence sequence");
		}

		if (futureTimeStamp <= evidenceSequence.length - 1) {
			throw new Error(`Future time stamp ${futureTimeStamp} is invalid for evidenceSequence length ${evidenceSequence.length}`);
		}

		const lookahead = futureTimeStamp - evidenceSequence.length + 1;

		const totalStates = this.transitionModel.length;
		const p: number[][] = this.defineBlank2DArray(lookahead, totalStates);

		const filteredResult = this.filtering(evidenceSequence);
		for (let t = 0; t < lookahead; ++t) {
			const previousKnowledge = t === 0? filteredResult[evidenceSequence.length - 1] : p[t - 1];

			for (let state = 0; state < totalStates; ++state) {

				p[t][state] = 0;
				for (let previousState = 0; previousState < totalStates; ++previousState) {
					p[t][state] += previousKnowledge[previousState] * this.transitionModel[previousState][state];
				}

			}

		}

		return p;
	}

	private fromStateToRemainingEvidence(evidenceSequence: number[], pastTimeStamp: number): number[][] {
		if (!this.evidenceSequenceIsValid(evidenceSequence)) {
			throw new Error("Invalid evidence sequence");
		}

		if (pastTimeStamp < 0 || pastTimeStamp >= evidenceSequence.length - 1) {
			throw new Error(`invalid past time stamp ${pastTimeStamp} for evidence sequence length ${evidenceSequence.length}`);
		}
		
		const totalStates = this.transitionModel.length;
		const b: number[][] = this.defineBlank2DArray(evidenceSequence.length - 1 - pastTimeStamp, totalStates);
		
		for (let t = evidenceSequence.length - 2; t >= pastTimeStamp; --t) {
			const nextEvidence = evidenceSequence[t + 1];
			const i = t - evidenceSequence.length + b.length + 1;
			const nextKnowledge: number[] = t === evidenceSequence.length - 2? new Array(totalStates).fill(1) : b[i + 1];
			for (let state = 0; state < totalStates; ++state) {
				b[i][state] = 0;
				for (let nextState = 0; nextState < totalStates; ++nextState) {
					b[i][state] += this.transitionModel[state][nextState] * this.sensorModel[nextState][nextEvidence] * nextKnowledge[nextState];
				}
			}
		}

		return b;
	}

	public smoothing(evidenceSequence: number[], pastTimeStamp: number): number[][] {
		if (!this.evidenceSequenceIsValid(evidenceSequence)) {
			throw new Error("Invalid evidence sequence");
		}

		if (pastTimeStamp < 0 || pastTimeStamp >= evidenceSequence.length - 1) {
			throw new Error(`invalid past time stamp ${pastTimeStamp} for evidence sequence length ${evidenceSequence.length}`);
		}

		const filteredResult = this.filtering(evidenceSequence);
		const fromStateToRemainingEvidence = this.fromStateToRemainingEvidence(evidenceSequence, pastTimeStamp);

		const totalStates = this.transitionModel.length;
		const s: number[][] = this.defineBlank2DArray(evidenceSequence.length - 1 - pastTimeStamp, totalStates);

		for (let t = pastTimeStamp; t < evidenceSequence.length - 1; ++t) {
			const i = t - pastTimeStamp;
			for (let state = 0; state < totalStates; ++state) {
				s[i][state] = filteredResult[t][state] * fromStateToRemainingEvidence[i][state];
			}
			s[i] = this.normalize(s[i]);
		}

		return s;
	}

	
	public overall(evidenceSequence: number[], fromTimeStamp: number, toTimeStamp: number): number[][] {
		if (!this.evidenceSequenceIsValid(evidenceSequence)) {
			throw new Error("Invalid evidence sequence");
		}

		if (fromTimeStamp > toTimeStamp) {
			throw new Error(`fromTimestamp ${fromTimeStamp} is greater than toTimeStamp ${toTimeStamp}`);
		}
		
		if (evidenceSequence.length - 1 < fromTimeStamp) {
			return this.prediction(evidenceSequence, toTimeStamp).slice(fromTimeStamp - evidenceSequence.length);
		}

		if (evidenceSequence.length - 1 > toTimeStamp) {
			return this.smoothing(evidenceSequence, fromTimeStamp).slice(0, toTimeStamp + 1);
		}

		const smoothing = fromTimeStamp < evidenceSequence.length - 1? this.smoothing(evidenceSequence, fromTimeStamp): [];
		const filtering = [this.filtering(evidenceSequence)[evidenceSequence.length - 1]];
		const prediction = toTimeStamp > evidenceSequence.length - 1? this.prediction(evidenceSequence, toTimeStamp) : [];

		return smoothing.concat(filtering).concat(prediction);
	}


	private normalize(array: number[]): number[] {
		const copy = array.slice();
		const total = copy.reduce((accumulator, currentValue) => accumulator + currentValue);
		return copy.map(value => value / total);
	}


	private evidenceSequenceIsValid(evidenceSequence: number[]): boolean {
		for (const evidence of evidenceSequence) {
			if (evidence < 0 || evidence >= this.sensorModel[0].length) return false;
		}
		return true;
	}


	private transitionModelIsValid(transitionModel: number[][]): boolean {
		let length = transitionModel.length;
		for (const row of transitionModel) {
			if (row.length !== length - 1) return false;
			const total = row.reduce((accumulator, currentValue) => accumulator + currentValue);
			if (total > 1) return false;
		}
		return true;
	}

	private sensorModelIsValid(sensorModel: number[][]): boolean {
		for (const row of sensorModel) {
			if (row.length !== sensorModel[0].length) return false;
			const total = row.reduce((accumulator, currentValue) => accumulator + currentValue);
			if (total > 1) return false;
		}
		return true;
	}

	private priorProbabilityIsValid(priorProbability: number[], numberOfStates: number) {
		if (priorProbability.length !== numberOfStates - 1) return false;
		const total = priorProbability.reduce((accumulator, currentValue) => accumulator + currentValue);
		return total <= 1;
	}

	private defineBlank2DArray(rows: number, cols: number): number[][] {
		const array: number[][] = new Array(rows);
		for (let i = 0; i < array.length; ++i) {
			array[i] = new Array(cols);
		}
		return array;
	}

}
