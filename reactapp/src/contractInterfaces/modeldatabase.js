import web3 from './web3';
import { modelDatabase} from "../abi/abi";

const address = '0xeeeD5f2549Da8c48f480acEf08b8974dFd3F8205';

export default new web3.eth.Contract(modelDatabase, address);
