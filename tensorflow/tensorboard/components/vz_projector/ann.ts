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

Refactored code from LargeVis: https://github.com/elbamos/largeVis
==============================================================================*/

import { Vector } from './vector';
import * as vector from './vector';
import { NearestEntry } from './knn';
import { DataPoint } from './data';
import { KMin } from './heap'

export class Position {
	constructor(matrix: NearestEntry[][], column: number) {

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
				let neighboriterator = 0;
				let it3 = 0;
				while(neighboriterator < tmp.length && it3 < indicesEnd) {
					let newone = indices[neighboriterator];
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
		let v = d.slice();
		vector.unit(v);
		for(let i = 0; i< I; i++) {
			const I2 = indices[i];
			const X = this.data[I2].vector;
			direction[i] = vector.dot(vector.sub(X, m), v);
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
			console.log('tree: ' + i)
		}

	}

	public reduce(): NearestEntry[][] {
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
		return this.knns;
	}

	public exploreNeighborhood(maxIter: number) {
	this.K = this.knns[0].length;
	let old_knns: NearestEntry[][] = [];
	for(let T = 0; T < maxIter; ++T) {
		old_knns = this.knns;
		let nodeHeap: NearestEntry[] = [];
		let positionHeap: NearestEntry[] = [];
		let positionVector:Position[] = [];
		for(let i = 0; i < this.N; ++i) {
			const x_i = this.data[i].vector;
			/*
				for (auto it = old_knns.begin_col(i); it != oldEnd; ++it) {
		if (*it == -1) break;
		positionVector.emplace_back(old_knns, *it);
		vertexidxtype id = * (positionVector.back().first);
		positionHeap.insert(posVecCnt++, id);
	}

	vertexidxtype lastOne = -1;
	while (! positionHeap.isEmpty()) {
		const vertexidxtype nextOne = positionHeap.minKey();

		if (nextOne != lastOne && nextOne != i) {
			addHeap(nodeHeap, x_i, nextOne);
			lastOne = nextOne;
		}
		advanceHeap(positionHeap, positionVector);
	}

	/*
	* Before the last iteration, we keep the matrix sorted by vertexid, which makes the merge above
	* more efficient.
	*
	* We can't use std:copy because we're copying from a vector of pairs

	auto copyContinuation = std::transform(nodeHeap.begin(), nodeHeap.end(), knns.begin_col(i),
										[](const std::pair<distancetype, vertexidxtype>& input) {return input.second;});
	if (copyContinuation == knns.begin_col(i)) throw Rcpp::exception("No neighbors after exploration - this is a bug.");
	sort(knns.begin_col(i), copyContinuation);
	std::fill(copyContinuation, knns.end_col(i), -1);*/
			}
		}

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
	public test(): Promise<NearestEntry[][]> {
		return new Promise<NearestEntry[][]>((resolve) => resolve(this.knns))
	}

}
