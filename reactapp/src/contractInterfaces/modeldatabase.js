import web3 from './web3';
import { modelDatabase} from "../abi/abi";

const address = '0x6AABC017bA7a784fF7e5F07BE93322b3735454Fe';

export default new web3.eth.Contract(modelDatabase, address);