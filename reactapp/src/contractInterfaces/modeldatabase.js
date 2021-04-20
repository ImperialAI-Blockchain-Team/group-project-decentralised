import web3 from './web3';
import { modelDatabase} from "../abi/abi";

const address = '0xa41f5792aa12bB8162A7e6f694E8bd9fC990C42c';

export default new web3.eth.Contract(modelDatabase, address);
