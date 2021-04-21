import web3 from './web3';
import { modelDatabase} from "../abi/abi";

const address = '0xC4cE2515dc8Fb8Bd6eC41076DEd2c39a55BcD727';

export default new web3.eth.Contract(modelDatabase, address);
