/* Copyright 2017 Ryangwook Ryoo. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Refactored code from: https://github.com/elbamos/largeVis
==============================================================================*/

import { Vector } from './vector';
import * as vector from './vector';
import { NearestEntry } from './knn';
import { DataPoint } from './data';
import { KMin } from './heap'
import {runAsyncTask} from './util';

export class Position {
	first: number;
	knn:NearestEntry[][];
	i: number;
	constructor(knn: NearestEntry[][], i: number) {
		this.knn = knn;
		this.first = 0;
		this.i = i;
	}
	public advance(): number {
		this.first = this.first + 1;
		return (this.first >= this.knn[this.i].length-1) ? -1 : this.knn[this.i][this.first].index;
	}
	public get(): number {
		return this.knn[this.i][this.first].index;
	}
}

export class MinIndexedPQ {
	private N: number;
	private heap: number[];
	private index: number[];
	private keys: number[];

	constructor(NMAX: number) {
		const nn = NMAX + 1;
		this.N = 0;
		this.heap = new Array<number>(nn);
		this.index = new Array<number>(nn);
		this.keys = new Array<number>(nn);
		for(let i = 0; i < nn; i++) {
			this.index[i] = -1;
		}
	}

	private swap(i:number, j: number): void {
		let tmp = this.heap[i];
		this.heap[i] = this.heap[j];
		this.heap[j] = tmp;
		this.index[this.heap[i]] = i;
		this.index[this.heap[j]] = j;
	}

	private bubbleUp(k: number): void {
		while(k > 1 && this.keys[this.heap[Math.floor(k/2)]] > this.keys[this.heap[k]]) {
			this.swap(k, Math.floor[k/2]);
			k = Math.floor(k/2);
		}
	}

	private bubbleDown(k:number): void {
		while(2*k <= this.N) {
			let j = 2*k;
			if(j < this.N && this.keys[this.heap[j]] > this.keys[this.heap[j+1]]) j++;
			if(this.keys[this.heap[k]] <= this.keys[this.heap[j]]) break;
			this.swap(k, j);
			k = j;
		}
	}

	public isEmpty(): boolean {
		return this.N == 0;
	}

	public insert(i:number, key:number): void {
		this.N = this.N + 1;
		this.index[i] = this.N;
		this.heap[this.N] = i;
		this.keys[i] = key;
		this.bubbleUp(this.N);
	}

	public minIndex(): number {
		return this.heap[1];
	}

	public minKey(): number {
		return this.keys[this.heap[1]];
	}

	public pop(): number {
		let min = this.heap[1];
		this.N = this.N - 1;
		this.swap(1, this.N);
		this.bubbleDown(1);
		this.index[min] = -1;
		this.heap[this.N+1] = -1;
		return min;

	}
	public rotate(key: number): void {
		let i = this.heap[1];
		this.keys[i] = key;
		this.bubbleDown(this.index[i]);
	}
}

/**
 * Compute KNN using Random Projection Trees
 */
export class ANN {
	private localNeighborhood: number[][];
	private treeNeighborhoods: number[][];
	private knns: NearestEntry[][];
	data: DataPoint[];
	K: number;
	N: number;
	threshold: number = 0;
	threshold2: number = 0;
	progress: number;

	constructor(data: DataPoint[], K:number) {
		this.data = data;
		this.K = K;
		this.N = data.length;
		this.treeNeighborhoods = [];
		for(let i = 0; i < this.N; i++) {
			this.treeNeighborhoods[i] = new Array<number>();
		}
	}

	private median(direction: Float32Array): number {
		let tmp = direction.slice();
		tmp.sort((a, b) => { return a-b });
		const i = tmp.length / 2;
		return i % 1 === 0 ? tmp[i-1] : (tmp[Math.floor(i)-1] + tmp[Math.floor(i)])/2;
	}

	private recurse(indices: number[]): void {
		const I = indices.length;
		if(I <= this.threshold) {
			this.localNeighborhood.push(indices);
		} else {
			let direction = this.hyperplane(indices);

			let middle = this.median(direction);
			let left:number[] = [];
			for(let i = 0; i < direction.length; i++) {
				if(direction[i] > middle) {
					left.push(indices[i])
				}
			}
			if(left.length > (I - 2) || left.length < 2) {
				for(let i = 0; i < direction.length ; i++) {
					direction[i] = Math.random();
				}
				middle = 0.5;
				left = [];
				for(let i = 0; i < direction.length; i++) {
					if(direction[i] > middle) {
						left.push(indices[i])
					}
				}
			}
			let right:number[] = [];
			for(let i = 0; i < direction.length; i++) {
				if(direction[i] <= middle) {
					right.push(indices[i])
				}
			}
			this.recurse(left);
			this.recurse(right);
		}
	}
	private mergeNeighbors(): void {
		let tmp:number[] = [];
		for(let it = 0; it < this.localNeighborhood.length ; ++it) {
			const indices = this.localNeighborhood[it];
			const indicesEnd = indices.length;
			for(let it2 = 0; it2 < indicesEnd; ++it2) {
				const cur = indices[it2];
				tmp = this.treeNeighborhoods[cur].slice();
				this.treeNeighborhoods[cur] = [];
				let neighboriterator = 0;
				let it3 = 0;
				while(neighboriterator < tmp.length && it3 < indicesEnd) {
					let newone = tmp[neighboriterator];
					if(newone < indices[it3]) {
						++neighboriterator;
					} else if (indices[it3] < newone) {
						if(indices[it3] === cur) {
							++it3;
							continue;
						}
						newone = indices[it3];
						++it3;
					} else {
						++neighboriterator;
						++it3;
					}
					this.treeNeighborhoods[cur].push(newone);
				}
				this.treeNeighborhoods[cur].concat(tmp.slice(neighboriterator));
				for(let i = it3; i < indicesEnd; i++) {
					if(indices[i] !== cur) {
						this.treeNeighborhoods[cur].push(indices[i]);
					}
				}
			}
		}
	}

	hyperplane(indices: number[]): Float32Array {
		const I = indices.length;
		let direction = new Float32Array(I);
		const x1idx = Math.random() * (I-1);
		let x2idx = Math.random() * (I-2);
		x2idx = (x2idx >= x1idx) ? (x2idx + 1) % I : x2idx;
		const x2 = this.data[indices[Math.round(x1idx)]].vector;
		const x1 = this.data[indices[Math.round(x2idx)]].vector;

		let m = new Float32Array(x1.length);
		for (let i = 0; i < x1.length; ++i) {
			m[i] = (x1[i] + x2[i]) / 2;
		}
		const d = vector.sub(x1, x2);
		if(I < this.threshold2 && vector.sum(d) === 0) {
			for(let i = 0; i < direction.length; i++) {
				direction[i] = Math.random();
			}
			return direction;
		}
		
		vector.unit(d);
		for(let i = 0; i< I; i++) {
			const I2 = indices[i];
			const X = this.data[I2].vector;
			direction[i] = vector.dot(vector.sub(X, m), d);
		}
		return direction;
	}
//  public setSeed
	public trees(n_trees: number, newThreshold: number): void {
		this.threshold = newThreshold;
		this.threshold2 = newThreshold * 4;
		let indices = [];
		for(let i = 0; i< this.data.length ; i++) {
			indices.push(i);
		}

		for(let i = 0; i < n_trees; i++) {
			this.localNeighborhood = [];
			this.recurse(indices);
			this.mergeNeighbors();
		}

	}

	public reduce(): void {
		this.knns = [];
		let newNeighborhood: NearestEntry[] = []; /* Max K*threshold of NearestEntry */

		for(let i = 0; i < this.N; ++i) {
			const x_i = this.data[i].vector;
			let neighborhood = this.treeNeighborhoods[i];
			let kMin = new KMin<NearestEntry>(this.K);
			for(let j = 0; j < neighborhood.length; ++j) {
				const d = vector.dist2(x_i, this.data[neighborhood[j]].vector);
				kMin.add(d, {index: neighborhood[j], dist: d});
			}

			this.knns[i] = kMin.getMinKItems();
		}
	}

	public exploreNeighborhood(maxIter: number): void {
	let old_knns: NearestEntry[][];
	for(let T = 0; T < maxIter; ++T) {
		old_knns = this.knns;
		let nodeHeap = new KMin<NearestEntry>(this.K);
		let positionHeap: MinIndexedPQ = new MinIndexedPQ(this.K + 1);
		let positionVector:Position[] = [];
		for(let i = 0; i < this.N; ++i) {
			const x_i = this.data[i].vector;
			positionVector.push(new Position(old_knns, i));
			positionHeap.insert(0, old_knns[i][0].index);
			let posVecCnt: number = 1;
			let oldEnd = old_knns[i].length;
			for(let it = 0; it < oldEnd; it++) {
				positionVector.push(new Position(old_knns, old_knns[i][it].index));
				let id: number = old_knns[i][positionVector[positionVector.length-1].first].index;
				positionHeap.insert(posVecCnt, id);
				posVecCnt = posVecCnt + 1;
			}
			let lastOne = -1;
			while(!positionHeap.isEmpty()) {
				let nextOne = positionHeap.minKey();
				if(nextOne != lastOne && nextOne != i) {
					const d = vector.dist2(x_i, this.data[nextOne].vector);
					nodeHeap.add(d, {index: nextOne, dist: d});
					lastOne = nextOne;

				}
				this.advanceHeap(positionHeap, positionVector);
			}
			this.knns[i] = nodeHeap.getMinKItems();
			}
		}

	}

	private advanceHeap(positionHeap: MinIndexedPQ, positionVector: Position[]): void {
		let whichColumn = positionHeap.minIndex();
		let adv = positionVector[whichColumn].advance();
		if(adv == -1) {
			positionHeap.pop();
		} else {
			positionHeap.rotate(adv);
		}

	}

	private addHeap(heap: NearestEntry[], x_i: Float32Array, j: number): void {

		let d = vector.dist2(x_i, this.data[j].vector);

	}

	public sortAndReturn(): NearestEntry[][] {
		for(let i = 0; i < this.N; ++i) {
			let holder = new Array<NearestEntry>(this.K);
			const x_i = this.data[i].vector;
			for(let j = 0; j > this.knns[i].length; j++) {
				holder.push({index: j, dist: vector.dist2(x_i, this.data[j].vector)})
			}
			this.knns[i] = holder.sort((a, b) => {return a.dist-b.dist});
		}
		return this.knns;
	}

	public run(): Promise<NearestEntry[][]> {
		const Msg = 'Finding nearest neighbors';
		let progressMsg = Msg + ' - Building trees..';
		return runAsyncTask<void>(progressMsg, () => {
			console.time("RP");
			console.time("ANN");
			this.trees(50, 50);
			return;
		}).then(() => {
			progressMsg = Msg + ' - Reducing Neighbors..';
			runAsyncTask<void>(progressMsg, () => {
				this.reduce();
				console.timeEnd("RP"); 
				return;
			})
		}).then(() => {
			return runAsyncTask<NearestEntry[][]>(progressMsg, () => {
				progressMsg = Msg + ' - Exploring Neighborhood..';
				//this.exploreNeighborhood(1);
				console.timeEnd("ANN"); 
				return this.knns;
			});
		})

	}

}
