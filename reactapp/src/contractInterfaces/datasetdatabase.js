import web3 from './web3';
import { datasetDatabase} from "../abi/abi";

const address = '0x33153Cc4c4417775209e8B59cf5D2c48fD9cBBcC';

export default new web3.eth.Contract(datasetDatabase, address);
