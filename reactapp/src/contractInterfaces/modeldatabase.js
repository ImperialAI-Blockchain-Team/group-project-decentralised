import web3 from './web3';
import { modelDatabase} from "../abi/abi";

const address = '0xBc25955e1927C59360Da8799C991620e525B16C1';

export default new web3.eth.Contract(modelDatabase, address);