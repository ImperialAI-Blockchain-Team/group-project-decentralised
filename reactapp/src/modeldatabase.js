import web3 from './web3';
import { modelDatabase} from "./abi/abi";

const address = '0x04c2e84278437a12b46f6355E40B5f39B2148127';

export default new web3.eth.Contract(modelDatabase, address);